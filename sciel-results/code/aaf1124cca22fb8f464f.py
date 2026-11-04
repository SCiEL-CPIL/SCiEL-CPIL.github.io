"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(pts) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    n = len(pts)
    min_area = float('inf')
    
    # Efficient vectorized computation
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            # Cross product of (pj - pi) and (pk - pi)
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area / hull_area

def _negative_score(params):
    """
    Objective function for optimization: negative of the score.
    """
    pts = params.reshape(14, 2)
    s = _score(pts)
    if s <= 0:
        return 1e9
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Strategy: Global optimization using Differential Evolution in 28D space.
    # This avoids local minima traps of gradient-free local methods like Nelder-Mead
    # and explores the full configuration space without symmetry constraints.
    
    # Bounds: points roughly within [-1.5, 1.5] to ensure convex hull area is reasonable
    # and not degenerate. The score is scale-invariant, so this is a search space choice.
    bounds = [(-1.5, 1.5)] * 28
    
    # DE parameters
    # popsize: 10 * 28 = 280 individuals? Too many for 150s if each eval is ~0.1ms?
    # _score is ~1-2ms. 280 evals per gen. 
    # maxiter: 50 gens -> 14000 evals. 14000 * 2ms = 28s.
    # We can afford more. Let's try popsize=50, maxiter=100.
    # 50*28=1400 evals/gen. 100 gens = 140k evals. 140k * 2ms = 280s. Too slow.
    
    # Let's be conservative: popsize=30, maxiter=50.
    # 30*28=840 evals/gen. 50 gens = 42k evals. 42k * 2ms = 84s. Safe.
    
    try:
        res = differential_evolution(
            _negative_score, 
            bounds, 
            maxiter=50, 
            popsize=30, 
            seed=42, 
            tol=1e-4,
            polish=True,  # Use L-BFGS-B for final polish
            polish_options={'maxiter': 100, 'maxfev': 200}
        )
        
        if res.fun < 1e8: # Valid result
            best_pts = res.x.reshape(14, 2)
            best_score = -res.fun
        else:
            # Fallback to a simple symmetric configuration if DE fails
            angles = np.linspace(0, 2*np.pi, 14, endpoint=False)
            best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
            best_score = _score(best_pts)
            
    except Exception as e:
        # Fallback to a simple symmetric configuration if DE fails
        angles = np.linspace(0, 2*np.pi, 14, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_pts)
        
    # Check time, if we have spare time, try a quick local refinement on the best point
    if time.time() < deadline - 10.0:
        from scipy.optimize import minimize
        try:
            res2 = minimize(_negative_score, best_pts.flatten(), method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 100})
            if res2.fun < -best_score:
                best_pts = res2.x.reshape(14, 2)
                best_score = -res2.fun
        except Exception:
            pass
            
    return best_pts
# EVOLVE-BLOCK-END
