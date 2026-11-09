"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # seconds; leave margin before 120s kill

# EVOLVE-BLOCK-START

def _pairwise_dists2(pts):
    """Return matrix of squared pairwise distances (N x N)."""
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sum(diff * diff, axis=2)

def _score(pts):
    """Compute the common radius r for a set of centers."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    # Boundary constraint: distance to nearest edge
    # Square is [-0.5, 0.5]^2
    dist_to_boundary = 0.5 - np.max(np.abs(pts), axis=1)
    min_boundary = np.min(dist_to_boundary)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    d2 = _pairwise_dists2(pts)
    # Zero out diagonal
    np.fill_diagonal(d2, np.inf)
    min_pair_d2 = np.min(d2)
    if min_pair_d2 <= 0:
        return 0.0
    min_pair_d = np.sqrt(min_pair_d2)
    r = min(min_pair_d / 2.0, min_boundary)
    return r

def _initial_grid():
    """Create an initial grid arrangement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _random_start():
    """Random start within the square."""
    return np.random.uniform(-0.5, 0.5, size=(N, 2))

def _refine_local(pts, max_iter=200):
    """
    Local refinement using gradient-free optimization.
    We maximize the minimum of (pairwise_dist/2, boundary_dist).
    Equivalent to minimizing the negative of that.
    """
    def objective(x):
        pts = x.reshape(N, 2)
        r = _score(pts)
        if r <= 0:
            return -1e10
        # We want to maximize r, so minimize -r
        # But _score is nonsmooth; use a smooth surrogate via soft-min
        # Actually, let's just use the raw score for Powell which can handle it
        return -r
    
    best_pts = pts.copy()
    best_r = _score(pts)
    
    # Use Powell method which doesn't need gradients
    try:
        res = minimize(objective, pts.flatten(), method='Powell',
                       options={'maxiter': max_iter, 'xtol': 1e-12, 'ftol': 1e-12})
        if res.fun < -best_r:  # i.e., -r < -best_r => r > best_r
            cand = res.x.reshape(N, 2)
            cand_r = _score(cand)
            if cand_r > best_r:
                best_pts = cand
                best_r = cand_r
    except Exception:
        pass
    
    return best_pts, best_r

def _refine_with_softmin(pts, max_iter=100):
    """
    Refine using a smooth surrogate: soft-min of the constraints.
    r = min(pairwise_dists/2, boundary_dists)
    softmin(a, b) = -logsumexp(-a, -b) / beta approximately
    """
    beta = 50.0  # higher = closer to true min
    
    def softmin_obj(x):
        pts = x.reshape(N, 2)
        d2 = _pairwise_dists2(pts)
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(np.maximum(d2, 0))
        pair_vals = d / 2.0
        bound_vals = 0.5 - np.max(np.abs(pts), axis=1)
        
        # Combine all constraints into a single vector
        all_vals = np.concatenate([pair_vals.ravel(), bound_vals.ravel()])
        # Filter out non-finite
        all_vals = all_vals[np.isfinite(all_vals)]
        if len(all_vals) == 0:
            return 1e10
        
        # Soft-min: -log(sum(exp(-beta * v))) / beta
        # For numerical stability, shift by min
        v_min = np.min(all_vals)
        shifted = all_vals - v_min
        log_sum = np.log(np.sum(np.exp(-beta * shifted)))
        soft_min_val = v_min - np.log(np.sum(np.exp(-beta * shifted))) / beta
        
        return -soft_min_val  # minimize negative soft-min = maximize soft-min
    
    best_pts = pts.copy()
    best_r = _score(pts)
    
    try:
        res = minimize(softmin_obj, pts.flatten(), method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10,
                                'adaptive': True})
        if res.fun < -best_r:
            cand = res.x.reshape(N, 2)
            cand_r = _score(cand)
            if cand_r > best_r:
                best_pts = cand
                best_r = cand_r
    except Exception:
        pass
    
    return best_pts, best_r

def construct():
    t_start = time.time()
    
    # Start with grid
    pts = _initial_grid()
    best_pts = pts.copy()
    best_r = _score(pts)
    
    # Phase 1: Local refinement from grid
    if time.time() - t_start < DEADLINE - 30:
        best_pts, best_r = _refine_local(best_pts, max_iter=150)
    
    # Phase 2: Try random restarts
    n_restarts = 0
    while time.time() - t_start < DEADLINE - 20:
        n_restarts += 1
        pts = _random_start()
        # Quick local refinement
        pts, r = _refine_local(pts, max_iter=80)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # Phase 3: Polish the best solution
    if time.time() - t_start < DEADLINE - 5:
        best_pts, best_r = _refine_local(best_pts, max_iter=300)
    
    # Phase 4: Final high-precision polish with soft-min
    if time.time() - t_start < DEADLINE - 2:
        best_pts, best_r = _refine_with_softmin(best_pts, max_iter=200)
    
    # Final check
    final_r = _score(best_pts)
    if final_r <= 0:
        # Fallback to grid
        return _initial_grid()
    
    return best_pts

# EVOLVE-BLOCK-END
