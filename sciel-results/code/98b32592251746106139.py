"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_MARGIN = 5.0  # seconds before the 150s limit


def _radius(pts):
    """Compute the largest common radius for given centres."""
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)

    # Distance to walls
    wall_dists = np.min(np.minimum(0.5 - pts, pts + 0.5), axis=1)
    min_wall = np.min(wall_dists)

    return min(min_pair / 2.0, min_wall)


def _soft_min_obj(pts_flat, alpha=20.0):
    """
    Objective: minimize -soft_min of all constraints.
    Constraints: pairwise distances >= 2r, wall distances >= r.
    We reformulate: maximize the minimum of (pairwise_dist/2, wall_dist).
    
    Using soft-min: f = -(1/alpha) * log(sum(exp(-alpha * x_i))) approximates min(x_i)
    So we minimize -f = (1/alpha) * log(sum(exp(-alpha * x_i)))
    
    But we need to be careful: x_i are the "margins" we want to maximize.
    Let's define:
      For each pair (i,j): margin = dist_ij / 2
      For each point i, each axis k: margin = 0.5 - |c_ik|
    
    We want to maximize the minimum margin.
    Equivalent to minimizing the soft-max of (-margins).
    
    Actually, let's use a different approach: minimize the sum of squared violations
    plus a term that encourages large minimum.
    
    Better: use the log-sum-exp trick on the negative margins.
    """
    pts = pts_flat.reshape(-1, 3)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
    np.fill_diagonal(dists, np.inf)
    
    # Margins from pairs
    pair_margins = dists / 2.0
    pair_margins = pair_margins[np.triu_indices(N, k=1)]
    
    # Wall margins
    wall_margins = np.min(np.minimum(0.5 - pts, pts + 0.5), axis=1)
    
    # All margins
    all_margins = np.concatenate([pair_margins, wall_margins])
    
    # Soft-min: (1/alpha) * log(sum(exp(-alpha * margin))) approximates min(margin)
    # We want to maximize min(margin), so minimize -soft_min = (1/alpha)*log(sum(exp(alpha*(-margin))))
    # But for numerical stability:
    neg_margins = -all_margins
    max_neg = np.max(neg_margins)
    if max_neg > 50:
        # Clip to avoid overflow
        max_neg = 50
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_margins - max_neg)))
    obj = log_sum_exp / alpha
    
    return obj


def _soft_min_grad(pts_flat, alpha=20.0):
    """Gradient of the soft-min objective."""
    pts = pts_flat.reshape(-1, 3)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
    np.fill_diagonal(dists, np.inf)
    
    # Margins
    pair_margins = dists / 2.0
    pair_margins_flat = pair_margins[np.triu_indices(N, k=1)]
    
    wall_margins = np.min(np.minimum(0.5 - pts, pts + 0.5), axis=1)
    
    all_margins = np.concatenate([pair_margins_flat, wall_margins])
    neg_margins = -all_margins
    
    # Soft-min gradient
    max_neg = np.max(neg_margins)
    if max_neg > 50:
        max_neg = 50
    exp_vals = np.exp(neg_margins - max_neg)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # These are the softmax weights
    
    # Gradient with respect to margins: d(obj)/d(margin_i) = -weights[i] * alpha / alpha = -weights[i]
    # Actually: obj = (1/alpha) * log(sum(exp(-alpha * margin)))
    # d(obj)/d(margin_i) = (1/alpha) * (-alpha * exp(-alpha*margin_i)) / sum(exp(-alpha*margin_j))
    # = -exp(-alpha*margin_i) / sum(exp(-alpha*margin_j))
    # = -weights[i]
    
    # Now we need d(margin)/d(pts)
    
    # For wall margins: margin_i = min_k (0.5 - |c_ik|)
    # This is non-differentiable at the min, but we can use subgradient
    # For simplicity, use the gradient of the soft-min over wall distances too
    
    # Let me reformulate: compute gradients numerically for wall, analytically for pairs
    
    # Actually, let's just use numerical gradient for the whole thing to be safe
    # But that's slow. Let me do analytical for pairs.
    
    grad = np.zeros_like(pts_flat)
    
    # Pair contributions
    # For pair (i,j), margin = dist_ij / 2
    # d(margin)/d(pts_i) = (pts_i - pts_j) / dist_ij
    # d(margin)/d(pts_j) = (pts_j - pts_i) / dist_ij
    
    # We have weights for each pair margin
    n_pairs = N * (N - 1) // 2
    pair_weights = weights[:n_pairs]
    
    # Build the pair index matrix
    ii, jj = np.triu_indices(N, k=1)
    
    for idx in range(n_pairs):
        i, j = ii[idx], jj[idx]
        w = pair_weights[idx]
        d = pts[i] - pts[j]
        dist = np.sqrt(np.sum(d ** 2) + 1e-12)
        if dist > 1e-10:
            grad[3*i:3*i+3] += w * d / dist
            grad[3*j:3*j+3] -= w * d / dist
    
    # Wall contributions
    wall_weights = weights[n_pairs:]
    for i in range(N):
        w = wall_weights[i]
        # margin_i = min_k (0.5 - |c_ik|)
        # Subgradient: for the active constraint
        abs_pts = np.abs(pts[i])
        active_k = np.argmin(0.5 - abs_pts)
        sign = np.sign(pts[i, active_k])
        if sign == 0:
            sign = 1
        grad[3*i + active_k] -= w * sign
    
    return grad


def _build_initial_grid():
    """Build a reasonable initial configuration."""
    # Use a simple grid as starting point
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _local_search(pts, max_iter=2000, alpha=15.0):
    """Run L-BFGS-B optimization from initial points."""
    x0 = pts.flatten()
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (3 * N)
    
    result = minimize(
        _soft_min_obj,
        x0,
        args=(alpha,),
        jac=None,  # Use numerical gradient for robustness
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    
    return result.x.reshape(-1, 3)


def _perturb(pts, sigma=0.05):
    """Add random perturbation."""
    return pts + np.random.randn(*pts.shape) * sigma


def _fix_boundaries(pts):
    """Clip points to be inside the cube."""
    return np.clip(pts, -0.5, 0.5)


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    best_pts = _build_initial_grid()
    best_radius = _radius(best_pts)
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different initial conditions
    n_restarts = 20
    max_iter_per_restart = 3000
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial points
        if restart == 0:
            init = best_pts.copy()
        elif restart == 1:
            init = _build_initial_grid()
        else:
            # Random initialization with some structure
            m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            init = grid[:N].copy()
            init += rng.standard_normal(init.shape) * 0.02
            init = _fix_boundaries(init)
        
        # Run optimization
        try:
            opt_pts = _local_search(init, max_iter=max_iter_per_restart, alpha=15.0)
            opt_pts = _fix_boundaries(opt_pts)
            opt_radius = _radius(opt_pts)
            
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_pts = opt_pts.copy()
        except Exception:
            continue
        
        # If we have time, do a second pass with tighter tolerance
        if time.time() < deadline - 30:
            try:
                opt_pts2 = _local_search(opt_pts, max_iter=max_iter_per_restart, alpha=25.0)
                opt_pts2 = _fix_boundaries(opt_pts2)
                opt_radius2 = _radius(opt_pts2)
                
                if opt_radius2 > best_radius:
                    best_radius = opt_radius2
                    best_pts = opt_pts2.copy()
            except Exception:
                continue
    
    return best_pts
