"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _score(pts):
    """Largest common radius for a set of centres."""
    if pts is None or len(pts) == 0:
        return 0.0
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    # distance to each of the 4 walls
    wall = np.minimum(
        np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0]),
        np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    )
    wall_min = wall.min()
    
    # nearest neighbour distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min()
    
    return min(wall_min, 0.5 * nn)

def _neg_score(pts):
    return -_score(pts)

def _soft_min_obj(x, beta=100.0):
    """
    Smooth differentiable objective: minimize the soft-minimum of all
    pairwise distances and wall distances.
    x: flat array of shape (2N,)
    Returns scalar value to minimize (negative of soft-min radius).
    """
    pts = x.reshape(N, 2)
    
    # Wall distances (4 per point)
    wx = pts[:, 0] + 0.5
    ww = 0.5 - pts[:, 0]
    wy = pts[:, 1] + 0.5
    wtt = 0.5 - pts[:, 1]
    wall_dists = np.concatenate([wx, ww, wy, wtt])
    
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, np.inf)
    pair_dists = d[np.triu_indices(N, k=1)]
    
    # All distances
    all_dists = np.concatenate([wall_dists, pair_dists])
    
    # Soft minimum using log-sum-exp trick:
    # softmin = -log(sum(exp(-d_i * beta))) / beta
    # We minimize -softmin, so we maximize softmin
    # For numerical stability, subtract min first
    d_min = np.min(all_dists)
    shifted = all_dists - d_min
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = d_min - np.log(np.sum(np.exp(-beta * shifted))) / beta
    
    return -soft_min

def _soft_min_grad(x, beta=100.0):
    """Gradient of the soft-min objective."""
    pts = x.reshape(N, 2)
    
    # Wall distances
    wx = pts[:, 0] + 0.5
    ww = 0.5 - pts[:, 0]
    wy = pts[:, 1] + 0.5
    wtt = 0.5 - pts[:, 1]
    wall_dists = np.concatenate([wx, ww, wy, wtt])
    
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2) + 1e-30)
    np.fill_diagonal(d, np.inf)
    pair_dists = d[np.triu_indices(N, k=1)]
    
    all_dists = np.concatenate([wall_dists, pair_dists])
    
    d_min = np.min(all_dists)
    shifted = all_dists - d_min
    exp_neg = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_neg)
    weights = exp_neg / sum_exp  # softmax weights over distances
    
    # Gradient of soft_min w.r.t. each distance: -w_i
    # Then chain rule through each distance to get gradient w.r.t. x
    
    grad = np.zeros_like(x)
    
    # Wall gradients
    # wx[i] = pts[i,0] + 0.5 -> d(wx)/d(pts[i,0]) = 1
    # ww[i] = 0.5 - pts[i,0] -> d(ww)/d(pts[i,0]) = -1
    # wy[i] = pts[i,1] + 0.5 -> d(wy)/d(pts[i,1]) = 1
    # wtt[i] = 0.5 - pts[i,1] -> d(wtt)/d(pts[i,1]) = -1
    
    w_weights = weights[:4*N]
    # For each point i, wall distances are at indices 4*i, 4*i+1, 4*i+2, 4*i+3
    for i in range(N):
        w0 = w_weights[4*i]      # wx
        w1 = w_weights[4*i+1]    # ww
        w2 = w_weights[4*i+2]    # wy
        w3 = w_weights[4*i+3]    # wtt
        # d(soft_min)/d(pts[i,0]) = -w0*1 + w1*(-1) = -(w0 - w1)
        # We want grad of -soft_min, so grad = +w0 - w1 for x-component
        grad[2*i] = -(w0 - w1)
        grad[2*i+1] = -(w2 - w3)
    
    # Pairwise distance gradients
    # For pair (i,j), distance d_ij = ||pts[i] - pts[j]||
    # d(d_ij)/d(pts[i]) = (pts[i] - pts[j]) / d_ij
    # d(d_ij)/d(pts[j]) = (pts[j] - pts[i]) / d_ij
    p_weights = weights[4*N:]
    
    # Build index mapping
    idx = np.triu_indices(N, k=1)
    pi = idx[0]
    pj = idx[1]
    
    for k in range(len(pi)):
        i, j = pi[k], pj[k]
        w = p_weights[k]
        diff_ij = pts[i] - pts[j]
        d_ij = np.linalg.norm(diff_ij)
        if d_ij < 1e-10:
            continue
        unit = diff_ij / d_ij
        # d(soft_min)/d(pts[i]) += -w * unit
        # grad of -soft_min: +w * unit for pts[i], -w * unit for pts[j]
        grad[2*i:2*i+2] += w * unit
        grad[2*j:2*j+2] -= w * unit
    
    return grad

def _make_hex_initial(jitter=0.0):
    """
    Create a hexagonal lattice configuration for 50 points.
    Uses alternating row pattern: [7, 6, 7, 6, 7, 6, 7, 4] = 50
    """
    row_lengths = [7, 6, 7, 6, 7, 6, 7, 4]
    n_rows = len(row_lengths)
    
    # Estimate spacing to fit in unit square
    # Width: max row has 7 circles, so 6 gaps -> need 6*s < 1.0
    # Height: 8 rows -> 7 gaps -> need 7 * s*sqrt(3)/2 < 1.0
    s_width = 0.9 / 6.0
    s_height = 0.9 / (7.0 * np.sqrt(3.0) / 2.0)
    spacing = min(s_width, s_height)
    
    v_spacing = spacing * np.sqrt(3.0) / 2.0
    
    pts = []
    total_height = (n_rows - 1) * v_spacing
    start_y = -total_height / 2.0
    
    for i, n in enumerate(row_lengths):
        y = start_y + i * v_spacing
        # Hex offset: alternate rows shift by half spacing
        offset = (spacing / 2.0) if (i % 2 == 1) else 0.0
        
        xs = np.arange(n) * spacing
        center = (xs[0] + xs[-1]) / 2.0 if n > 1 else 0.0
        xs = xs - center + offset
        
        for x in xs:
            pts.append([x, y])
    
    pts = np.array(pts, dtype=float)
    
    # Add jitter if requested
    if jitter > 0:
        np.random.seed(42)
        pts += np.random.uniform(-jitter, jitter, pts.shape)
    
    return pts

def _refine_lbfgs(pts, t0, deadline, beta=100.0):
    """
    Refine points using L-BFGS-B with soft-minimum objective.
    """
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    x0 = best_pts.flatten()
    
    try:
        res = minimize(
            _soft_min_obj,
            x0,
            method='L-BFGS-B',
            jac=_soft_min_grad,
            args=(beta,),
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        new_pts = res.x.reshape(N, 2)
        if np.all(np.isfinite(new_pts)):
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
    except Exception:
        pass
    
    return best_pts, best_score

def _refine_powell(pts, t0, deadline):
    """
    Refine points using Powell method on the discrete score.
    """
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    try:
        res = minimize(_neg_score, best_pts.flatten(), method="Powell",
                       options={'maxiter': 300, 'ftol': 1e-10, 'xtol': 1e-10, 'disp': False})
        
        new_pts = res.x.reshape(N, 2)
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts
    except Exception:
        pass
    
    return best_pts, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin for 150s limit
    
    # Phase 0: Generate initial hex lattice
    best_pts = _make_hex_initial(jitter=0.0)
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        np.random.seed(42)
        best_pts = np.random.uniform(-0.4, 0.4, (N, 2))
    
    best_score = _score(best_pts)
    
    # Phase 1: L-BFGS-B refinement with soft-minimum
    if time.time() < deadline - 20.0:
        best_pts, best_score = _refine_lbfgs(best_pts, t0, deadline, beta=100.0)
    
    # Phase 2: Powell refinement on discrete score
    if time.time() < deadline - 10.0:
        best_pts, best_score = _refine_powell(best_pts, t0, deadline)
    
    # Phase 3: Another L-BFGS-B pass with higher beta for sharper min
    if time.time() < deadline - 5.0:
        best_pts, best_score = _refine_lbfgs(best_pts, t0, deadline, beta=200.0)
    
    # Phase 4: Final Powell polish
    if time.time() < deadline - 2.0:
        best_pts, best_score = _refine_powell(best_pts, t0, deadline)
    
    return best_pts.reshape(N, 2)

# EVOLVE-BLOCK-END
