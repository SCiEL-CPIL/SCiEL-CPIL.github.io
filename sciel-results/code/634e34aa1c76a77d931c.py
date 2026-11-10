"""Improved packing for 50 equal circles in a unit square [-1/2, 1/2]^2."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50

def _score(pts):
    """Compute the max radius for a set of points (N,2)."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check bounds: all points must be strictly inside or on the boundary
    # The margin from each point to each wall
    margins = np.minimum(
        np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]),
        np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
    )
    wall_r = np.min(margins)
    if wall_r <= 0:
        return 0.0
    # Pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(pts, metric='euclidean')
    min_pair = np.min(d)
    if min_pair <= 0:
        return 0.0
    return min(min_pair / 2.0, wall_r)

def _neg_score(pts):
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    best_pts = None
    best_score = 0.0
    
    def try_update(pts):
        nonlocal best_pts, best_score
        pts = np.asarray(pts, dtype=float).reshape(N, 2)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Initial guess: 7x7 grid with some removed, or 5x10, etc.
    # Try several grid-like initializations
    # 5x10 grid
    def make_grid(cols, rows):
        xs = np.linspace(-0.5 + 1.0/cols, 0.5 - 1.0/cols, cols)
        ys = np.linspace(-0.5 + 1.0/rows, 0.5 - 1.0/rows, rows)
        gx, gy = np.meshgrid(xs, ys, indexing='xy')
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        return pts[:N]
    
    # 5x10 = 50 exactly
    init_pts = make_grid(5, 10)
    try_update(init_pts)
    
    # 7x7 = 49, add one
    def make_grid_7x7():
        m = 7
        xs = (np.arange(m) + 0.5) / m - 0.5
        ys = (np.arange(m) + 0.5) / m - 0.5
        gx, gy = np.meshgrid(xs, ys, indexing='xy')
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        # Add a 50th point in the center
        center = np.array([[0.0, 0.0]])
        pts = np.vstack([pts, center])
        return pts[:N]
    
    init_pts2 = make_grid_7x7()
    try_update(init_pts2)
    
    # 6x9 = 54, take 50
    init_pts3 = make_grid(6, 9)
    try_update(init_pts3)
    
    # Start with the best initial
    if best_pts is None:
        best_pts = init_pts.copy()
    
    # Nelder-Mead refinement from multiple starts
    from scipy.optimize import OptimizeResult
    np.random.seed(42)
    
    def nelder_mead_refine(start_pts, maxiter=300):
        x0 = start_pts.ravel()
        res = minimize(_neg_score, x0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-12})
        return res.x.reshape(N, 2)
    
    # Refine from the best initial
    if time.time() < deadline:
        refined = nelder_mead_refine(best_pts, maxiter=500)
        try_update(refined)
    
    # Try differential evolution with bounded search
    # This is expensive; use it if time permits
    if time.time() < deadline - 30:
        try:
            bounds = [(-0.5, 0.5)] * (2 * N)
            # Use a subset of popsize to save time
            de_res = differential_evolution(
                _neg_score, bounds, maxiter=50, popsize=15,
                tol=1e-8, seed=42, polish=False,
                init='latinhypercube'
            )
            de_pts = de_res.x.reshape(N, 2)
            try_update(de_pts)
            
            # Refine DE result
            if time.time() < deadline - 5:
                refined_de = nelder_mead_refine(de_pts, maxiter=300)
                try_update(refined_de)
        except Exception:
            pass
    
    # Additional local refinement passes
    if time.time() < deadline - 10 and best_pts is not None:
        for _ in range(3):
            if time.time() >= deadline - 10:
                break
            refined = nelder_mead_refine(best_pts, maxiter=200)
            try_update(refined)
    
    # Final safety: ensure result is valid
    if best_pts is not None:
        # Clamp to be strictly inside
        best_pts = np.clip(best_pts, -0.4999999, 0.4999999)
        # Re-verify
        s = _score(best_pts)
        if s > 0:
            return best_pts
        else:
            # Fallback to grid
            return init_pts[:N]
    else:
        return init_pts[:N]
