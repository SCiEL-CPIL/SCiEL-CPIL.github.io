"""
Heilbronn convex regions: Maximize minimum triangle area / hull area.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import itertools

N = 13

# EVOLVE-BLOCK-START
import random

def _compute_score(pts):
    """
    Calculate the score: min triangle area / convex hull area.
    pts: (N, 2) array.
    """
    # Check for degenerate hull
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-10:
        return 0.0

    # Calculate all triangle areas
    # Number of triangles is C(N, 3)
    # We can vectorize this partially or just loop. N=13 is small.
    n = len(pts)
    min_area = float('inf')
    
    # Vectorized approach for triangle areas
    # For each pair (i, j), compute area with all k > j? No, just all triples.
    # Let's use a simple loop for clarity and correctness, N=13 is fast enough.
    # 13 choose 3 = 286 triangles.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(Pj-Pi, Pk-Pi)|
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area

    return min_area / hull_area

def _objective(x):
    """
    Objective function for minimization.
    x: flattened array of the free coordinates.
    We fix P0=(0,0) and P1=(1,0).
    x contains coordinates for P2...P12 (11 points, 22 values).
    """
    # Reconstruct points
    pts = np.zeros((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    free_pts = x.reshape(-1, 2)
    pts[2:] = free_pts
    
    score = _compute_score(pts)
    # We want to maximize score, so minimize -score
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial random points to help DE or just use DE bounds
    # Bounds for free points: Since we fixed scale/rot/trans, 
    # the other points should be in a reasonable range.
    # Let's try bounds [-2, 2] for x and y.
    bounds = [(-2.0, 2.0)] * (2 * (N - 2))
    
    # Differential Evolution
    # maxiter and popsize need to be balanced with time.
    # 22 dimensions. Popsize 10-20. Maxiter 100-200.
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15, 
            tol=1e-6, 
            atol=1e-8,
            init='sobol',
            polish=False # We will polish manually to control time
        )
    except Exception:
        # Fallback to random if DE fails
        pass
    else:
        best_x = res.x
        best_score = -res.fun
        
        # Polish the best result
        # Use Nelder-Mead for robustness against non-smoothness
        try:
            res_polish = minimize(
                _objective, 
                best_x, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res_polish.fun < res.fun:
                best_x = res_polish.x
                best_score = -res_polish.fun
        except Exception:
            pass
        
        # Check time
        if time.time() < deadline - 5.0:
            # Try a few random restarts if time permits?
            # Or just return the best found.
            # Let's try 2 more quick DE runs with different seeds if time allows.
            # But DE is expensive. Let's stick to the one DE + polish.
            
            # Reconstruct final points
            pts = np.zeros((N, 2))
            pts[0] = [0.0, 0.0]
            pts[1] = [1.0, 0.0]
            pts[2:] = best_x.reshape(-1, 2)
            
            return pts

    # Fallback: Random points
    rng = np.random.default_rng(42)
    pts = rng.random((N, 2))
    return pts

# EVOLVE-BLOCK-END
