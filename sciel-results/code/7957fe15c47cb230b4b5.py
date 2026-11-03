"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

def _hull_area(pts):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _min_tri_area(pts):
    """Calculate the minimum area of any triangle formed by three points."""
    n = len(pts)
    if n < 3:
        return 0.0
    
    # Get indices for all combinations i < j < k
    idx = np.array(list(itertools.combinations(range(n), 3)))
    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    # Vectorized cross product calculation
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _score(pts):
    """Calculate the score: min triangle area / hull area."""
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 0.0
    mta = _min_tri_area(pts)
    return mta / ha

def _objective_flat(flat_pts):
    """
    Objective function for the optimizer.
    Minimizes the negative of the score.
    Normalizes points to unit hull area to remove scale invariance.
    """
    pts = flat_pts.reshape(N, 2)
    ha = _hull_area(pts)
    if ha < 1e-8:
        return 1e6  # Penalty for degenerate configurations
    
    # Normalize to unit area
    scale = np.sqrt(1.0 / ha)
    pts_norm = pts * scale
    
    mta = _min_tri_area(pts_norm)
    return -mta

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    best_score = -np.inf
    best_pts = None
    
    # Generate initial guesses
    guesses = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    guesses.append(reg_pts)
    
    # 2. Random points in circle
    for i in range(5):
        rng = np.random.default_rng(i)
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        guesses.append(pts)
        
    # 3. Random points in square
    for i in range(5):
        rng = np.random.default_rng(100 + i)
        pts = rng.random((N, 2)) * 2 - 1
        guesses.append(pts)
        
    # 4. Grid-like structure
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    grid_pts = np.array([[xi, yi] for xi in x for yi in y])
    guesses.append(grid_pts[:13])

    for guess in guesses:
        if time.time() > deadline:
            break
            
        # Optimize
        try:
            res = minimize(
                _objective_flat, 
                guess.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
            )
            opt_pts = res.x.reshape(N, 2)
            score = _score(opt_pts)
            
            if score > best_score:
                best_score = score
                best_pts = opt_pts
        except Exception:
            continue
            
    # Fallback if no optimization succeeded
    if best_pts is None:
        best_pts = guesses[0]
        
    return best_pts
# EVOLVE-BLOCK-END
