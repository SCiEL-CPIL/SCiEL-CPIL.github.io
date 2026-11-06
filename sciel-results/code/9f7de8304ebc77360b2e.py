"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centers):
    """Compute the valid common radius for a set of centers in the unit cube [-0.5, 0.5]^3."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check centers are within cube
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    # Distance to walls
    wall_dists = np.min(np.minimum(centers + 0.5, 0.5 - centers), axis=1)
    min_wall = np.min(wall_dists)
    # Pairwise distances
    # Use broadcasting for pairwise distances (N=100 is small enough)
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    r = min(min_wall, min_pair)
    if r <= 0 or not np.isfinite(r):
        return 0.0
    return r


def _soft_min(values, beta=50.0):
    """Approximate the minimum using log-sum-exp."""
    # We want min, so use -max(-values)
    v = -values
    v_max = np.max(v)
    if v_max > 100:
        return -v_max
    return -(v_max + np.log(np.sum(np.exp(v - v_max))))


def _objective(x, beta=50.0):
    """
    Objective to minimize: -radius + penalty for constraint violations.
    We want to maximize r = min(min_pair/2, min_wall).
    Using soft-min to make it differentiable.
    """
    centers = x.reshape(-1, 3)
    
    # Wall distances
    wall_dists = np.min(np.minimum(centers + 0.5, 0.5 - centers), axis=1)
    
    # Pairwise distances / 2
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Combine all constraints into a single array for soft-min
    # We want the minimum of all wall distances and all pair distances
    # Flatten pair distances (excluding diagonal)
    mask = ~np.eye(N, dtype=bool)
    flat_pair = pair_dists[mask]
    
    all_vals = np.concatenate([wall_dists, flat_pair])
    
    # Soft min
    r_soft = -_soft_min(-all_vals, beta)
    
    # Penalty for hard violations (optional, but soft-min should handle it)
    # Add a large penalty if any value is negative (overlap or outside)
    penalty = 0.0
    if np.min(all_vals) < 0:
        penalty = 100.0 * np.sum(np.clip(-all_vals, 0, None) ** 2)
    
    return -r_soft + penalty


def _objective_grad(x, beta=50.0):
    """
    Gradient of the objective.
    """
    centers = x.reshape(-1, 3)
    grad = np.zeros_like(x)
    
    # Wall distances
    # d/dx_i (min(centers_i + 0.5, 0.5 - centers_i))
    # This is min of two linear functions. Gradient is subgradient.
    # Let's use the soft-min approach consistently.
    
    # Recompute all_vals
    wall_dists = np.min(np.minimum(centers + 0.5, 0.5 - centers), axis=1)
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    mask = ~np.eye(N, dtype=bool)
    flat_pair = pair_dists[mask]
    all_vals = np.concatenate([wall_dists, flat_pair])
    
    # Soft min gradient:
    # f(v) = -logsumexp(-v)
    # df/dv_i = exp(-v_i) / sum(exp(-v_j))
    v = -all_vals
    v_max = np.max(v)
    exps = np.exp(v - v_max)
    weights = exps / np.sum(exps)
    
    # Gradient of -f(v) w.r.t v is -weights
    # But we want gradient of objective = -r_soft + penalty
    # r_soft = -logsumexp(-all_vals)
    # d(r_soft)/d(all_vals)_i = weights_i
    # d(objective)/d(all_vals)_i = -weights_i
    
    # Now propagate to centers
    # Wall part
    for i in range(N):
        w = weights[i]
        # wall_dists[i] = min(centers[i] + 0.5, 0.5 - centers[i])
        # This is a min of linear functions.
        # If centers[i] + 0.5 < 0.5 - centers[i] (i.e., centers[i] < 0), 
        # then wall_dist = centers[i] + 0.5, grad = (1, 1, 1)
        # If centers[i] >= 0, wall_dist = 0.5 - centers[i], grad = (-1, -1, -1)
        # To be smooth, we can use a soft-min here too, or just use the active one.
        # For simplicity in gradient descent, let's use the active constraint.
        # Actually, let's just compute the gradient of the min function properly.
        # min(a, b) gradient is a subgradient. We'll pick the one with smaller value.
        a = centers[i] + 0.5
        b = 0.5 - centers[i]
        if np.min(a) < np.min(b):
            # Active is a = centers[i] + 0.5
            # grad of min(a) w.r.t centers[i] is the index of min
            idx = np.argmin(a)
            grad[i * 3 + idx] += -w * 1.0
        else:
            idx = np.argmin(b)
            grad[i * 3 + idx] += -w * (-1.0)
            
    # Pair part
    # flat_pair corresponds to upper triangle of pair_dists
    # pair_dists[i,j] = 0.5 * ||c_i - c_j||
    # d(pair_dists[i,j])/d(c_i) = 0.5 * (c_i - c_j) / ||c_i - c_j||
    # d(pair_dists[i,j])/d(c_j) = 0.5 * (c_j - c_i) / ||c_i - c_j||
    
    # Iterate over pairs
    for i in range(N):
        for j in range(i + 1, N):
            idx_flat = i * N + j - (i * (i + 1) // 2 + j - 1) # This is getting complicated.
            # Let's just recompute the index in flat_pair
            # The mask is ~np.eye(N). The order is row-major.
            # For i < j, the index in the flattened array of the upper triangle...
            # Actually, np.fill_diagonal and then mask=~eye gives all off-diagonal.
            # The order is: (0,1), (0,2), ..., (0,N-1), (1,0), ...
            # This is not just upper triangle.
            # Let's just compute the gradient contribution directly.
            
            # Find the weight for this pair in the all_vals array.
            # This is inefficient. Let's just use numerical gradient or a simpler approach.
            # Given the complexity, let's stick to numerical gradient for the first version
            # or use a simpler objective.
            pass
            
    # For the sake of time, let's just use numerical gradient in the optimizer
    # or simplify the objective.
    
    # Actually, let's just return None and let scipy use numerical gradient.
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial guess: random points
    np.random.seed(42)
    best_centers = None
    best_radius = 0.0
    
    # Try a few random restarts
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Random initialization
        centers = np.random.uniform(-0.4, 0.4, (N, 3))
        
        # Flatten for optimizer
        x0 = centers.flatten()
        
        # Optimize
        try:
            res = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 100000}
            )
            
            if res.success or res.status in [1, 2, 3]:
                opt_centers = res.x.reshape(-1, 3)
                # Clip to cube to ensure validity
                opt_centers = np.clip(opt_centers, -0.5, 0.5)
                
                r = _compute_radius(opt_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = opt_centers.copy()
        except Exception:
            pass
            
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
