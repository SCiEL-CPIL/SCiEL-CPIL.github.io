"""Improved sphere packing: diverse grid inits + high-beta polish with deadline guard."""
N = 100
import time
import numpy as np

# EVOLVE-BLOCK-START
import numpy as np

def _compute_radius(pts):
    """Compute the common radius for a set of points in [-0.5,0.5]^3."""
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    if n == 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    # Distance to boundary
    dist_bound = np.min(0.5 - np.abs(pts), axis=1)
    min_bound = np.min(dist_bound)
    return 0.5 * min(min_pair, 2.0 * min_bound)

def _score(pts):
    """Score = radius."""
    return _compute_radius(pts)

def _grid_init(m1, m2, m3):
    """Create a grid of m1*m2*m3 points, take first N."""
    x = (np.arange(m1) + 0.5) / m1 - 0.5
    y = (np.arange(m2) + 0.5) / m2 - 0.5
    z = (np.arange(m3) + 0.5) / m3 - 0.5
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return pts[:N]

def _random_init(seed):
    """Random initialization inside the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))

def _polish(pts, beta_max, time_budget):
    """
    High-beta polish: maximize the minimum pair distance using a smooth surrogate.
    Uses log-sum-exp soft-min for the pair distances and a smooth boundary term.
    """
    pts = pts.astype(np.float64).copy()
    n = pts.shape[0]
    t0 = time.time()
    
    # Build pair index arrays once (excluding diagonal)
    i_idx, j_idx = np.where(np.triu(np.ones((n, n), dtype=bool), k=1))
    n_pairs = len(i_idx)
    
    # Initial beta
    beta = 10.0
    lr = 0.001
    
    for step in range(5000):
        if time.time() - t0 > time_budget:
            break
        
        # Compute pair distances
        diff = pts[i_idx] - pts[j_idx]
        d = np.sqrt(np.sum(diff * diff, axis=1))
        
        # Soft-min of pair distances via log-sum-exp
        # log(sum(exp(-beta*d))) / (-beta) approximates min(d)
        # For numerical stability:
        d_min = np.min(d)
        exp_terms = np.exp(-beta * (d - d_min))
        log_sum = np.log(np.sum(exp_terms)) + (-beta * d_min)
        soft_min_d = -log_sum / beta
        
        # Gradient of soft_min_d w.r.t. pts
        # d(soft_min_d)/d(pts[i]) = sum_j w_ij * (pts[i] - pts[j]) / d_ij
        # where w_ij = exp(-beta*d_ij) / sum(exp(-beta*d_ij))
        w = exp_terms / np.sum(exp_terms)
        
        grad = np.zeros_like(pts)
        # For each pair (i, j): contribution to grad[i] is w * (pts[i]-pts[j])/d
        # contribution to grad[j] is -w * (pts[i]-pts[j])/d
        diff_unit = diff / np.maximum(d, 1e-15)[:, None]
        contrib = w[:, None] * diff_unit
        np.add.at(grad, i_idx, contrib)
        np.add.at(grad, j_idx, -contrib)
        
        # Boundary term: we want to maximize min(0.5 - |pts|)
        # Soft-min of boundary distances
        bound_d = 0.5 - np.abs(pts)  # shape (n, 3)
        bd_min = np.min(bound_d)
        bd_exp = np.exp(-beta * (bound_d - bd_min))
        bd_log_sum = np.log(np.sum(bd_exp)) + (-beta * bd_min)
        soft_min_b = -bd_log_sum / beta
        
        # Gradient of soft_min_b w.r.t. pts
        # d(soft_min_b)/d(pts[i,k]) = -sign(pts[i,k]) * w_b[i,k]
        # where w_b[i,k] = exp(-beta*bound_d[i,k]) / sum(exp(-beta*bound_d))
        w_b = bd_exp / np.sum(bd_exp)
        grad_b = -np.sign(pts) * w_b
        
        # Combine gradients: maximize soft_min_d + soft_min_b
        # But we need to weight them appropriately
        # The radius is min(0.5*min_pair, min_bound)
        # So gradient direction: 0.5 * grad_d + grad_b (roughly)
        total_grad = 0.5 * grad + grad_b
        
        # Update
        pts += lr * total_grad
        
        # Clamp to stay in cube (with small margin)
        pts = np.clip(pts, -0.499, 0.499)
        
        # Adapt beta and learning rate
        beta *= 1.05
        if step % 100 == 0:
            lr *= 0.9
        
        if beta > beta_max:
            beta = beta_max
            lr *= 0.8
    
    return pts

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin
    
    # Generate diverse initializations
    inits = []
    # 5x4x5 = 100
    inits.append(_grid_init(5, 4, 5))
    # 6x5x4 = 120, take 100
    inits.append(_grid_init(6, 5, 4))
    # 4x5x5 = 100
    inits.append(_grid_init(4, 5, 5))
    # 5x5x4 = 100
    inits.append(_grid_init(5, 5, 4))
    # 4x4x7 = 112, take 100
    inits.append(_grid_init(4, 4, 7))
    # Random starts
    for s in [42, 123, 456]:
        inits.append(_random_init(s))
    
    best_pts = None
    best_score = -1.0
    
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Quick polish
        polished = _polish(init_pts, beta_max=50000, time_budget=15.0)
        sc = _score(polished)
        
        if sc > best_score:
            best_score = sc
            best_pts = polished.copy()
        
        # If we have time, do a second polish from the best
        if time.time() < deadline - 20.0 and idx < len(inits) - 1:
            continue
    
    # Final high-precision polish on the best configuration
    if best_pts is not None and time.time() < deadline - 10.0:
        remaining = max(5.0, deadline - time.time() - 2.0)
        best_pts = _polish(best_pts, beta_max=100000, time_budget=remaining)
        final_score = _score(best_pts)
        if final_score > best_score:
            best_score = final_score
    
    if best_pts is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    # Final validation and clamping
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_pts
# EVOLVE-BLOCK-END
