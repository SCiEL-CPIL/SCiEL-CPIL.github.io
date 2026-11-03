"""
Circle packing: 200 equal circles in unit square [-0.5, 0.5]^2.
Uses scipy L-BFGS-B with a smooth objective combining pairwise repulsion
and boundary repulsion, plus multi-start from hexagonal and random inits.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
DEADLINE = 145.0  # wall-clock safety margin (evaluator kills at 150s)


def _compute_radius(pts):
    """Compute the maximum common radius for the given centres."""
    # Distance to boundary
    boundary_min = np.min(0.5 - np.abs(pts))
    # Pairwise distances
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    r = min(min_pair, 2.0 * boundary_min) / 2.0
    return r


def _soft_min(dists, t):
    """Soft minimum of an array of distances using log-sum-exp."""
    # dists is 1D array of positive distances
    min_val = np.min(dists)
    # Avoid overflow: shift by min
    shifted = dists - min_val
    # soft min = min_val - (1/t) * log(sum(exp(-t * shifted)))
    # For large t, this approaches min_val
    exp_vals = np.exp(-t * shifted)
    log_sum = np.log(np.sum(exp_vals))
    return min_val - log_sum / t


def _objective_and_grad(x, t=50.0):
    """
    Objective: minimize -soft_min(pairwise_distances, t) - soft_min(boundary_dists, t)
    Actually we want to maximize the minimum distance, so minimize the negative.
    
    x: flattened array of shape (2*N,)
    Returns (f, g) where f is objective value and g is gradient.
    """
    pts = x.reshape(-1, 2)
    
    # Pairwise distances
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    
    # Boundary distances: for each point, min over 4 walls
    # dist to left wall: pts[:,0] + 0.5
    # dist to right wall: 0.5 - pts[:,0]
    # dist to bottom wall: pts[:,1] + 0.5
    # dist to top wall: 0.5 - pts[:,1]
    bound_dists = np.stack([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ], axis=1)  # shape (N, 4)
    # For each point, the min boundary distance
    # We'll use soft min over all boundary distances (4*N values)
    bound_flat = bound_dists.flatten()
    
    # We want to maximize min(min_pair_dist, min_boundary_dist)
    # Use soft min over pairwise distances (excluding diagonal) and boundary distances
    # Combine: soft min of [all pairwise dists, all boundary dists]
    pair_flat = d.flatten()
    # Remove diagonal (inf values)
    pair_flat = pair_flat[pair_flat < np.inf]
    
    all_dists = np.concatenate([pair_flat, bound_flat])
    
    # Soft min
    min_val = np.min(all_dists)
    shifted = all_dists - min_val
    exp_vals = np.exp(-t * shifted)
    log_sum = np.log(np.sum(exp_vals))
    soft_min_val = min_val - log_sum / t
    
    # Objective: negative of soft min (we want to maximize)
    f = -soft_min_val
    
    # Gradient
    # d(soft_min)/d(all_dists[i]) = -exp(-t*(all_dists[i]-min_val)) / sum(exp(-t*(all_dists[j]-min_val)))
    weights = exp_vals / np.sum(exp_vals)
    
    grad_all = -weights  # df/d(all_dists) = -d(soft_min)/d(all_dists) = weights... wait
    # f = -soft_min
    # df/d(all_dists[i]) = -d(soft_min)/d(all_dists[i])
    # d(soft_min)/d(all_dists[i]) = -exp(-t*(all_dists[i]-min_val)) / sum(exp(...)) * (-1) ... 
    # Let me recompute: soft_min = min_val - (1/t) * log(sum(exp(-t*(d_i - min_val))))
    # d(soft_min)/d(d_k) = 1 - (1/t) * d(log(sum(exp(-t*(d_i-min_val)))))/d(d_k)
    # d(log(S))/d(d_k) where S = sum(exp(-t*(d_i-min_val)))
    # = (1/S) * d(S)/d(d_k)
    # d(S)/d(d_k) = -t * exp(-t*(d_k-min_val))  (if k is in the set)
    # So d(soft_min)/d(d_k) = 1 - (1/t) * (-t * exp(-t*(d_k-min_val)) / S)
    # = 1 + exp(-t*(d_k-min_val)) / S
    # Hmm, this doesn't look right. Let me use a cleaner formulation.
    
    # Alternative: use the fact that soft_min ≈ min for large t
    # The gradient of soft_min w.r.t. each d_i is:
    # exp(-t*(d_i - min_val)) / sum_j exp(-t*(d_j - min_val))
    # This is a probability distribution that concentrates on the minimum.
    
    # So d(soft_min)/d(d_i) = w_i where w_i = exp(-t*(d_i-min_val)) / sum(exp(-t*(d_j-min_val)))
    # And f = -soft_min, so df/d(d_i) = -w_i
    
    # Now we need df/dx where x is the positions.
    # For pairwise distances: d_ij = |p_i - p_j|
    # dd_ij/dp_i = (p_i - p_j) / d_ij
    # dd_ij/dp_j = (p_j - p_i) / d_ij
    
    # For boundary distances:
    # d_left_i = p_i[0] + 0.5, dd/dp_i[0] = 1
    # d_right_i = 0.5 - p_i[0], dd/dp_i[0] = -1
    # etc.
    
    grad_x = np.zeros_like(x)
    
    # Pairwise contribution
    # weights for pairwise: first N*(N-1) entries in weights
    n_pairs = len(pair_flat)
    pair_weights = weights[:n_pairs]
    bound_weights = weights[n_pairs:]
    
    # Reconstruct pairwise distance matrix for gradient computation
    # d is (N, N) with inf on diagonal
    # pair_flat was d.flatten() with diagonal removed
    # The order is row-major: d[0,1], d[0,2], ..., d[0,N-1], d[1,0], d[1,2], ...
    
    idx = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            w = pair_weights[idx]
            dist = d[i, j]
            if dist > 1e-12:
                diff = pts[i] - pts[j]
                dir_vec = diff / dist
                # df/dp_i += -w * dir_vec
                # df/dp_j += -w * (-dir_vec) = w * dir_vec
                grad_x[2*i:2*i+2] += -w * dir_vec
                grad_x[2*j:2*j+2] += w * dir_vec
            idx += 1
    
    # Boundary contribution
    # bound_flat order: for each point i, [left, right, bottom, top]
    for i in range(N):
        w_left = bound_weights[4*i]
        w_right = bound_weights[4*i+1]
        w_bottom = bound_weights[4*i+2]
        w_top = bound_weights[4*i+3]
        
        # df/dp_i[0] += -w_left * 1 + -w_right * (-1) = -w_left + w_right
        # df/dp_i[1] += -w_bottom * 1 + -w_top * (-1) = -w_bottom + w_top
        grad_x[2*i] += -w_left + w_right
        grad_x[2*i+1] += -w_bottom + w_top
    
    return f, grad_x


def _hex_init(n, rng):
    """Initialize n points in a hexagonal lattice pattern."""
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    rows = int(np.ceil(1.0 / dy)) + 2
    cols = int(np.ceil(1.0 / dx)) + 2
    pts = []
    for i in range(rows):
        y = -0.5 + i * dy
        offset = 0.0 if i % 2 == 0 else dx / 2.0
        for j in range(cols):
            x = -0.5 + j * dx + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append((x, y))
    pts = np.array(pts)
    if len(pts) >= n:
        idx = rng.choice(len(pts), size=n, replace=False)
        return pts[idx]
    else:
        result = np.empty((n, 2))
        result[:len(pts)] = pts
        result[len(pts):] = rng.uniform(-0.45, 0.45, size=(n - len(pts), 2))
        return result


def _random_init(n, rng):
    """Random initialization inside the unit square."""
    return rng.uniform(-0.45, 0.45, size=(n, 2))


def _grid_init(n, rng):
    """Grid initialization with slight random jitter."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= n:
        pts = grid[:n]
    else:
        pts = grid
        extra = rng.uniform(-0.4, 0.4, size=(n - len(pts), 2))
        pts = np.vstack([pts, extra])
    # Add small jitter
    pts = pts + rng.normal(0, 0.01, size=pts.shape)
    pts = np.clip(pts, -0.49, 0.49)
    return pts


def construct():
    """Construct 200 circle centres in the unit square maximizing common radius."""
    start_time = time.time()
    deadline = start_time + DEADLINE - 3.0  # 3 second safety margin
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_r = -np.inf
    
    # Multi-start: try different initializations
    inits = []
    inits.append(_hex_init(N, rng))
    inits.append(_random_init(N, rng))
    inits.append(_grid_init(N, rng))
    inits.append(_hex_init(N, rng))
    inits.append(_random_init(N, rng))
    
    for start_idx, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
        
        x0 = init_pts.flatten()
        
        try:
            res = minimize(_objective_and_grad, x0, method='L-BFGS-B',
                           jac=True,
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': 300, 'gtol': 1e-6, 'maxfun': 5000,
                                    'ftol': 1e-8, 'maxls': 50})
            
            if res.x is not None and np.all(np.isfinite(res.x)):
                pts_opt = res.x.reshape(-1, 2)
                # Clip to ensure validity (strictly inside)
                pts_opt = np.clip(pts_opt, -0.4999, 0.4999)
                
                r = _compute_radius(pts_opt)
                
                if r > best_r:
                    best_r = r
                    best_pts = pts_opt.copy()
                    
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    # If no valid result, return a simple grid
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_pts
