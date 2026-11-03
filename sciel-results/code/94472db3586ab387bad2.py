import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 13
TIME_LIMIT = 145.0 # Safety margin

def _compute_areas(pts):
    """
    Compute areas of all triangles formed by the points.
    pts: (N, 2) array
    Returns: array of shape (C(N,3),)
    """
    n = len(pts)
    areas = np.zeros(len(list(combinations(range(n), 3))))
    idx = 0
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            for k in range(j + 1, n):
                pk = pts[k]
                # Area = 0.5 * |det(pj-pi, pk-pi)|
                v1 = pj - pi
                v2 = pk - pi
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                areas[idx] = 0.5 * abs(cross)
                idx += 1
    return areas

def _compute_hull_area(pts):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        # If points are collinear or identical, area is 0
        return 0.0

def _score(pts):
    """
    Compute the score: min triangle area / hull area.
    """
    try:
        hull_area = _compute_hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
        areas = _compute_areas(pts)
        min_area = np.min(areas)
        return min_area / hull_area
    except Exception:
        return 0.0

def _objective(x):
    """
    Objective function for minimization.
    x: flat array of length 2*N
    """
    pts = x.reshape(-1, 2)
    return -_score(pts)

def _polish(pts, time_limit, start_time):
    """
    Local polish using Nelder-Mead.
    """
    x0 = pts.flatten()
    remaining = time_limit - (time.time() - start_time)
    if remaining < 5:
        return pts, _score(pts)
    
    # Nelder-Mead is good for non-smooth functions
    try:
        res = minimize(
            _objective, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        best_pts = res.x.reshape(-1, 2)
        best_score = _score(best_pts)
        
        s0 = _score(pts)
        if s0 > best_score:
            best_pts = pts
            best_score = s0
            
        return best_pts, best_score
    except Exception:
        return pts, _score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Define bounds for differential evolution
    # We can normalize the search space. Let's say points are in [-1, 1]
    bounds = [(-1, 1)] * (2 * N)
    
    best_pts = None
    best_score = -1
    
    # Run Differential Evolution
    # We might run it multiple times if time permits, or just once with a good population
    # popsize=15, maxiter depends on time
    
    # Estimate time per evaluation. 
    # _score involves ConvexHull (O(N log N)) and Area calc (O(N^3) but N=13 is small).
    # Let's say 1 eval takes ~0.001s. 
    # DE with popsize 15, maxiter 100 -> 1500 evals -> 1.5s.
    # We have 145s. We can afford much more.
    
    # Strategy: Run DE for a significant portion of the time, then polish.
    
    try:
        # Use a seed for reproducibility
        # Differential evolution is stochastic.
        
        # We can increase popsize and maxiter significantly.
        # Let's aim for ~10000 evaluations.
        # popsize=20, maxiter=500 -> 10000 evals.
        
        de_result = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=20, 
            tol=1e-6,
            polish=False, # We will do our own polish
            init='sobol' # Sobol sequence for better initial distribution
        )
        
        if de_result.success or de_result.fun < 0:
            cand_pts = de_result.x.reshape(-1, 2)
            cand_score = -de_result.fun
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand_pts
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed < deadline - 10:
            # Polish the best result found
            if best_pts is not None:
                polished_pts, polished_score = _polish(best_pts, deadline, start_time)
                if polished_score > best_score:
                    best_score = polished_score
                    best_pts = polished_pts
            else:
                # Fallback to a regular polygon if DE failed
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
                best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
                best_score = _score(best_pts)
        else:
            if best_pts is None:
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
                best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
                best_score = _score(best_pts)

    except Exception as e:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_pts)

    # If we have more time, run another DE instance with a different seed
    elapsed = time.time() - start_time
    if elapsed < deadline - 20 and best_pts is not None:
        try:
            de_result2 = differential_evolution(
                _objective, 
                bounds, 
                seed=123, 
                maxiter=100, 
                popsize=15, 
                tol=1e-6,
                polish=False
            )
            if de_result2.success or de_result2.fun < 0:
                cand_pts2 = de_result2.x.reshape(-1, 2)
                cand_score2 = -de_result2.fun
                if cand_score2 > best_score:
                    best_score = cand_score2
                    best_pts = cand_pts2
                    
                    # Polish the new best
                    elapsed = time.time() - start_time
                    if elapsed < deadline - 5:
                        polished_pts2, polished_score2 = _polish(best_pts, deadline, start_time)
                        if polished_score2 > best_score:
                            best_score = polished_score2
                            best_pts = polished_pts2
        except Exception:
            pass

    return best_pts
