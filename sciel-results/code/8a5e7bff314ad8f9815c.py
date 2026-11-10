"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: evaluator kills after 150s, use 120s as our limit
_DEADLINE = 120.0
_START_TIME = None


def _compute_radius(centres):
    """Compute the radius: min of half min-pair-distance and min wall distance."""
    if centres.shape[0] < 2:
        return 0.0
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Wall distance
    wall_dists = np.min(np.minimum(0.5 - np.abs(centres), axis=1), axis=1)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)


def _soft_min_radius(centres, beta=50.0):
    """
    Smooth surrogate for the minimum radius.
    r = min(min_pair/2, min_wall)
    We use a log-sum-exp style soft-min for both components.
    """
    n = centres.shape[0]
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Soft-min of pairwise distances (we want the minimum, so use -log-sum-exp of -d)
    # But for numerical stability, work with squared distances
    # min_pair_dist = sqrt(min(dists_sq))
    # soft_min of dists = sqrt(soft_min of dists_sq) approximately
    
    # For soft-min of values v_i: soft_min = -(1/beta) * log(sum(exp(-beta * v_i)))
    # For distances: we want min of sqrt(d_i^2), which is sqrt(min(d_i^2))
    # soft_min of d_i = -(1/beta) * log(sum(exp(-beta * d_i)))
    
    # Use squared distances for the soft min to avoid sqrt
    # min(d_i) = sqrt(min(d_i^2))
    # soft_min(d_i) ≈ -(1/beta) * log(sum(exp(-beta * d_i)))
    
    # But computing exp(-beta * d_i) for all pairs might be expensive
    # Let's use a different approach: soft-min of squared distances
    # min_sq = min(d_i^2)
    # soft_min_sq = -(1/beta) * log(sum(exp(-beta * d_i^2)))
    # Then soft_min_dist = sqrt(soft_min_sq)
    
    # For numerical stability with large beta, clip the exponents
    min_sq = np.min(dists_sq)
    exp_args = -beta * (dists_sq - min_sq)
    exp_args = np.clip(exp_args, -50, 50)
    soft_min_sq = min_sq - np.log(np.mean(np.exp(exp_args))) / beta
    
    # Wall distances
    wall_dists = np.min(np.minimum(0.5 - np.abs(centres), axis=1), axis=1)
    min_wall = np.min(wall_dists)
    exp_args_w = -beta * (wall_dists - min_wall)
    exp_args_w = np.clip(exp_args_w, -50, 50)
    soft_min_wall = min_wall - np.log(np.mean(np.exp(exp_args_w))) / beta
    
    # Combine: soft_min of (soft_min_sq/2, soft_min_wall)
    # Actually, r = min(sqrt(soft_min_sq)/2, soft_min_wall)
    # Let's just take the min of the two soft values
    soft_pair = np.sqrt(soft_min_sq) / 2.0
    return min(soft_pair, soft_min_wall)


def _soft_min_radius_full(centres, beta=30.0):
    """
    A smoother combined objective. We want to maximize r = min(min_pair/2, min_wall).
    Use a soft-min over all constraints simultaneously.
    """
    n = centres.shape[0]
    # Pairwise
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Wall
    wall_dists = np.min(np.minimum(0.5 - np.abs(centres), axis=1), axis=1)
    
    # We want to maximize min( sqrt(d_i^2)/2 for all pairs, wall_j for all points )
    # Let's define values:
    # v_k = d_k / 2 for pairs, and w_j = wall_j for walls
    # We want soft_max of these? No, we want to maximize the minimum.
    # So we want a soft_min that is differentiable.
    
    # Collect all "constraint values"
    # For pairs: d_i / 2
    # For walls: w_j
    
    # Use log-sum-exp for soft-min:
    # soft_min(v) = -(1/beta) * log( sum(exp(-beta * v)) )
    
    # To avoid underflow, shift by min
    pair_vals = np.sqrt(dists_sq) / 2.0
    # Replace inf with large number
    pair_vals = np.where(np.isinf(pair_vals), 1000.0, pair_vals)
    
    all_vals = np.concatenate([pair_vals.flatten(), wall_dists])
    
    min_v = np.min(all_vals)
    exp_args = -beta * (all_vals - min_v)
    exp_args = np.clip(exp_args, -50, 50)
    soft_min = min_v - np.log(np.mean(np.exp(exp_args))) / beta
    
    return soft_min


def _hilbert_initialization(n=50):
    """
    Initialize points using a space-filling curve approach.
    Map 1D indices to 3D coordinates in [-0.5, 0.5]^3.
    """
    # Simple approach: use a grid-like mapping with some jitter
    # For 50 points, use a 4x4x4 grid (64 points) and take 50
    m = int(np.ceil(n ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Add small random jitter to break symmetry
    rng = np.random.default_rng(42)
    grid = grid[:n].copy()
    jitter = rng.normal(0, 0.01, size=grid.shape)
    grid = grid + jitter
    
    # Clip to valid range
    grid = np.clip(grid, -0.45, 0.45)
    return grid


def _optimize_from_initial(centres, time_budget=30.0):
    """
    Optimize the configuration using gradient-based methods.
    Maximize the soft-min radius.
    """
    global _START_TIME
    start = time.time()
    initial_time = time.time() - _START_TIME
    remaining = _DEADLINE - initial_time - 5.0  # 5s safety margin
    if remaining < 5.0:
        return centres
    
    # Use L-BFGS-B to maximize soft_min_radius
    # We minimize -soft_min_radius
    
    def objective(x):
        c = x.reshape(-1, 3)
        return -_soft_min_radius_full(c, beta=30.0)
    
    def gradient(x):
        c = x.reshape(-1, 3)
        # Numerical gradient (could be expensive)
        eps = 1e-7
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        return grad
    
    # Actually, let's use a simpler approach: scipy minimize with numerical gradient
    # But that's very slow for 150 variables.
    
    # Alternative: use a custom gradient for the soft-min
    
    def objective_and_grad(x):
        c = x.reshape(-1, 3)
        n = c.shape[0]
        beta = 30.0
        
        # Pairwise
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        pair_vals = np.sqrt(dists_sq) / 2.0
        pair_vals = np.where(np.isinf(pair_vals), 1000.0, pair_vals)
        
        # Wall
        wall_dists = np.min(np.minimum(0.5 - np.abs(c), axis=1), axis=1)
        
        all_vals = np.concatenate([pair_vals.flatten(), wall_dists])
        
        min_v = np.min(all_vals)
        exp_args = -beta * (all_vals - min_v)
        exp_args = np.clip(exp_args, -50, 50)
        weights = np.exp(exp_args)
        weights /= np.sum(weights)
        
        soft_min = min_v - np.log(np.mean(weights)) / beta
        
        # Gradient of soft_min w.r.t. each coordinate
        # d(soft_min)/d(v_k) = -weights[k] / beta
        # Then chain rule through v_k
        
        grad_x = np.zeros_like(x)
        
        # For pair values: v_k = |c_i - c_j| / 2
        # d(v_k)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
        # d(v_k)/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|)
        
        # For wall values: v_k = min over axes of (0.5 - |c_i|)
        # This is non-differentiable at the min, but we can use the subgradient
        
        # Let's compute gradient contributions
        # Pair contributions
        for i in range(n):
            for j in range(i + 1, n):
                d = c[i] - c[j]
                dist = np.sqrt(np.sum(d ** 2))
                if dist < 1e-10:
                    continue
                # Index in pair_vals.flatten()
                # pair_vals is n x n, flattened
                idx_i = i * n + j
                idx_j = j * n + i
                w_i = weights[idx_i]
                w_j = weights[idx_j]
                grad_pair = (w_i + w_j) / (2 * beta) * d / dist
                grad_x[i * 3:i * 3 + 3] += grad_pair
                grad_x[j * 3:j * 3 + 3] -= grad_pair
        
        # Wall contributions
        for i in range(n):
            # v_i = min over k of (0.5 - |c_ik|)
            # The gradient is -sign(c_ik) / beta * weight for the active axis
            # For simplicity, use the axis with minimum value
            wall_vals_per_axis = 0.5 - np.abs(c[i])
            min_axis = np.argmin(wall_vals_per_axis)
            w_wall = weights[n * n + i]  # Index in all_vals
            grad_wall = -np.sign(c[i, min_axis]) * w_wall / beta
            grad_x[i * 3 + min_axis] += grad_wall
        
        # We are maximizing soft_min, so minimize -soft_min
        # Gradient of -soft_min is -grad(soft_min)
        # But we computed grad of soft_min w.r.t. x
        # So objective gradient = -grad_x
        
        return -soft_min, -grad_x
    
    # Use L-BFGS-B
    x0 = centres.flatten()
    
    # Define bounds
    bounds = [(-0.5, 0.5)] * (3 * n)
    
    result = minimize(
        objective_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'maxfun': 500, 'gtol': 1e-6}
    )
    
    if result.success or result.fun < objective(x0):
        centres_opt = result.x.reshape(-1, 3)
        # Clip to valid range
        centres_opt = np.clip(centres_opt, -0.5, 0.5)
        return centres_opt
    else:
        return centres


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Initial placement
    centres = _hilbert_initialization(N)
    
    # Optimize
    centres = _optimize_from_initial(centres)
    
    # Check time
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - 10.0:
        # Try another optimization round with different beta
        def objective2(x):
            c = x.reshape(-1, 3)
            return -_soft_min_radius_full(c, beta=50.0)
        
        x0 = centres.flatten()
        bounds = [(-0.5, 0.5)] * (3 * N)
        
        try:
            result = minimize(
                objective2,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'maxfun': 300, 'gtol': 1e-7}
            )
            if result.fun < objective2(x0):
                centres = result.x.reshape(-1, 3)
                centres = np.clip(centres, -0.5, 0.5)
        except Exception:
            pass
    
    # Final validation and clipping
    centres = np.clip(centres, -0.5, 0.5)
    
    # Ensure no NaN or Inf
    if not np.all(np.isfinite(centres)):
        centres = _hilbert_initialization(N)
    
    return centres
# EVOLVE-BLOCK-END
