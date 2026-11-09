# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    n = len(points)
    if n < 2:
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    return (d_min / d_max) ** 2

def _objective(x, n=14, d=3):
    """Objective function for minimization (negative of score)."""
    points = x.reshape(n, d)
    return -_score(points)

def _make_icosahedron_plus_2():
    """Icosahedron (12 verts) + 2 points at poles of a face."""
    phi = (1 + np.sqrt(5)) / 2
    verts = []
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            verts.append([0, s1, s2 * phi])
            verts.append([s1, s2 * phi, 0])
            verts.append([s2 * phi, 0, s1])
    points = np.array(verts, dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Add 2 points: place them at the centers of two opposite faces
    # Face center of (0,1,phi), (0,-1,phi), (1,phi,0) etc.
    # Actually, let's just add 2 points at (0,0,1) and (0,0,-1) initially
    # The optimizer will move them
    p1 = np.array([0.0, 0.0, 1.0])
    p2 = np.array([0.0, 0.0, -1.0])
    points = np.vstack([points, p1, p2])
    return points

def _make_cuboctahedron_plus_2():
    """Cuboctahedron (12 verts) + 2 points."""
    verts = []
    for x in [1, -1]:
        for y in [1, -1]:
            verts.append([x, y, 0])
            verts.append([x, 0, y])
            verts.append([0, x, y])
    points = np.array(verts, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([-1.0, 0.0, 0.0])
    points = np.vstack([points, p1, p2])
    return points

def _make_random_unit(n=14, d=3, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    points = rng.randn(n, d)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 590  # 630s limit, keep 40s margin
    
    best_score = -1.0
    best_points = None
    
    # Generate diverse initial guesses
    guesses = []
    guesses.append(_make_icosahedron_plus_2())
    guesses.append(_make_cuboctahedron_plus_2())
    
    # Add random unit sphere points with various seeds
    for i in range(8):
        guesses.append(_make_random_unit(n, d, seed=1000 + i * 137))
    
    # Optimize each guess
    for i, guess in enumerate(guesses):
        if time.time() > deadline - 15:
            break
            
        x0 = guess.flatten()
        current_score = -_objective(x0)
        if current_score > best_score:
            best_score = current_score
            best_points = guess.copy()
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            if not np.any(np.isnan(res.x)):
                res_score = -res.fun
                if res_score > best_score:
                    best_score = res_score
                    best_points = res.x.reshape(n, d)
        except Exception:
            pass
    
    # If we still have time, do a final refinement from the best point
    if best_points is not None and time.time() < deadline - 10:
        x0 = best_points.flatten()
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if not np.any(np.isnan(res.x)):
                res_score = -res.fun
                if res_score > best_score:
                    best_score = res_score
                    best_points = res.x.reshape(n, d)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = _make_icosahedron_plus_2()
    
    # Ensure finite and correct shape
    if not np.all(np.isfinite(best_points)):
        best_points = _make_icosahedron_plus_2()
    
    return best_points

# EVOLVE-BLOCK-END
