import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

def _score(points):
    # points: (N, 2)
    # 1. Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area < 1e-9:
            return 0.0
    except:
        return 0.0
    
    # 2. Min Triangle Area
    min_area = float('inf')
    n = len(points)
    # Optimization: Only need to check triples? Yes.
    # Vectorized approach for speed:
    # For each i, j, compute area with all k?
    # Let's just use a simple loop, it's fast enough for N=14.
    for i in range(n):
        pi = points[i]
        for j in range(i+1, n):
            pj = points[j]
            # Vectorized over k
            # Area = 0.5 * | (pj - pi) x (pk - pi) |
            # pk - pi is points[:, None] - pi[None, :] ? No, points - pi
            # Let's do: v1 = pj - pi
            # v2 = points - pi
            # cross = v1[0]*v2[:,1] - v1[1]*v2[:,0]
            # areas = 0.5 * abs(cross)
            # Exclude i and j from k?
            # The area is 0 for k=i or k=j. We want min non-zero?
            # The problem says "min area over all triples". Usually implies distinct points.
            # If k=i, area is 0. So min area would be 0.
            # Wait, "triples" usually means distinct indices i, j, k.
            # So we must exclude k in {i, j}.
            
            v1 = pj - pi
            v2 = points - pi
            cross = v1[0]*v2[:,1] - v1[1]*v2[:,0]
            areas = 0.5 * np.abs(cross)
            
            # Mask out i and j
            areas[i] = float('inf')
            areas[j] = float('inf')
            
            current_min = np.min(areas)
            if current_min < min_area:
                min_area = current_min
                
    return min_area / hull_area

def _negative_score(points):
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    p0 = np.column_stack((x, y))
    
    best_points = p0.copy()
    best_score = _score(p0)
    
    # Local Search
    # Use Nelder-Mead
    # We can run multiple restarts if time permits
    
    # To improve convergence, we can slightly perturb the initial guess
    rng = np.random.default_rng(42)
    
    # Run optimization
    # maxiter is a guess, we will check time
    # Nelder-Mead can be slow. Let's try a few iterations.
    # Or use Powell which might be faster for this dimension.
    
    # Let's try a simple coordinate descent or Nelder-Mead.
    # Given the high dimension (28), Nelder-Mead might be okay.
    
    # We will run the optimizer for a fixed duration.
    # Since minimize doesn't have a time limit, we wrap it or use a small maxiter.
    # 14 points * 2 dims = 28 variables.
    # Nelder-Mead does ~N+1 function evaluations per iteration.
    # 29 evals/iter.
    # Each eval is ~ms.
    # We can do thousands of iterations in 140s.
    
    # Let's run Nelder-Mead with a reasonable maxiter, say 5000, and check time.
    # Actually, we can just run it and break if time exceeds.
    # But minimize doesn't support easy breaking.
    # We can run it in a loop of smaller chunks?
    # Or just trust that 5000 iterations is fast enough.
    # 5000 * 29 * 1ms = 145s. Too risky.
    # Let's estimate eval time.
    # ConvexHull on 14 pts is very fast.
    # Min triangle area loop is fast.
    # Let's say 0.1 ms per eval.
    # 5000 * 29 * 0.1 ms = 14.5 s. Safe.
    
    # Let's try 20000 iterations.
    # 20000 * 29 * 0.1 ms = 58 s. Safe.
    
    try:
        res = minimize(_negative_score, p0, method='Nelder-Mead', options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < -best_score: # res.fun is negative score, so smaller is better
            best_points = res.x
            best_score = -res.fun
    except Exception as e:
        pass
        
    # Check time
    if time.time() < deadline - 10.0:
        # Try a second restart with noise
        p1 = p0 + rng.normal(0, 0.1, size=(N, 2))
        try:
            res2 = minimize(_negative_score, p1, method='Nelder-Mead', options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res2.fun < -best_score:
                best_points = res2.x
                best_score = -res2.fun
        except Exception as e:
            pass

    return best_points
