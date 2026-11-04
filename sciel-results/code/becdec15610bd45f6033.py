import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 14

# Precompute indices for all triples
def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

_TRIPLE_INDICES = _get_triple_indices(N)

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    # Vectorized calculation
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    # Cross product in 2D
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _neg_score(points):
    """Negative score for minimization."""
    return -_score(points)

def _generate_ring_points(params, n_out, n_in):
    """
    Generate points from ring parameters.
    params: [r1, theta1, r2, theta2, ...]
    For 2 rings: [r_in, theta_offset]
    For 3 rings: [r1, theta1, r2, theta2] (outer fixed at 1.0)
    """
    points = []
    
    # Outer ring is always on unit circle
    # We need to distribute n_out points on outer ring
    # and n_in points on inner rings
    
    # For the 2-ring case (7+7):
    # params = [r, theta]
    # Outer: 7 points at angles 0, 2pi/7, ...
    # Inner: 7 points at angles theta, theta+2pi/7, ... with radius r
    
    # For the 3-ring case (e.g., 6+4+4 or 5+5+4):
    # This is more complex. Let's stick to 2-ring primarily, 
    # but allow for a "center" point effectively (r=0).
    
    # Let's define a generic generator based on n_out, n_in and params length
    # If len(params) == 2: 2 rings (n_out on unit, n_in on r)
    # If len(params) == 4: 3 rings (n1 on unit, n2 on r1, n3 on r2)
    
    if len(params) == 2:
        r, theta = params
        angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
        outer_pts = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
        
        angles_in = theta + np.linspace(0, 2*np.pi, n_in, endpoint=False)
        inner_pts = r * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
        
        return np.vstack([outer_pts, inner_pts])
        
    elif len(params) == 4:
        r1, theta1, r2, theta2 = params
        # Assume split: n_out is split into n1 (outer) and rest? 
        # No, the signature _generate_ring_points(params, n_out, n_in) implies fixed counts.
        # But for 3 rings, we need 3 counts. 
        # Let's restrict the 4-param search to specific splits like 6,4,4 or 5,5,4.
        # To keep it simple and robust, let's only use 2-ring search (7,7) 
        # and maybe (8,6) or (6,8) with 2 params.
        # The inspiration used DE on 2-param spaces.
        
        # Let's assume the 4-param case is not used in the main loop for simplicity, 
        # or we implement a specific 3-ring generator.
        # For now, return empty or raise to avoid bugs.
        raise NotImplementedError("3-ring generation not implemented in this combine step")

def _score_ring_2params(params, n_out, n_in):
    """
    Score for a 2-ring configuration defined by [r, theta].
    """
    try:
        points = _generate_ring_points(params, n_out, n_in)
        return _score(points)
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin for 120s limit
    
    best_score = 0
    best_points = None
    
    # 1. Explore 2-ring configurations using Differential Evolution
    # We test different splits of 14 points into two concentric regular polygons.
    # Splits: (7,7), (8,6), (6,8), (9,5), (5,9), etc.
    # (n_out, n_in) where n_out + n_in = 14
    
    splits = [(7, 7), (8, 6), (6, 8), (9, 5), (5, 9), (10, 4), (4, 10), (11, 3), (3, 11), (12, 2), (2, 12), (13, 1), (1, 13)]
    
    for n_out, n_in in splits:
        if time.time() > deadline - 10:
            break
            
        # Define bounds for DE
        # r in [0, 1], theta in [0, 2*pi/gcd(n_out, n_in)]
        g = np.gcd(n_out, n_in)
        theta_max = 2 * np.pi / g
        bounds = [(0.0, 1.0), (0.0, theta_max)]
        
        # Objective function for DE (minimize negative score)
        def objective(p):
            return -_score_ring_2params(p, n_out, n_in)
        
        # Run Differential Evolution
        # Keep population and iterations modest to fit in time budget
        # 13 splits * ~5-10s each = ~65-130s. Might be tight.
        # Let's reduce popsize/maxiter for later splits if time is running low.
        
        elapsed = time.time() - start_time
        remaining = deadline - 5 - elapsed
        
        if remaining < 2:
            break
            
        # Adjust DE parameters based on remaining time
        if remaining > 30:
            popsize = 15
            maxiter = 50
        elif remaining > 15:
            popsize = 10
            maxiter = 30
        else:
            popsize = 5
            maxiter = 10
            
        try:
            res = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-4,
                polish=True
            )
            
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = _generate_ring_points(res.x, n_out, n_in).copy()
                    
        except Exception:
            pass
            
    # 2. Refine the best found ring configuration with Nelder-Mead on the full 28D space
    # This allows breaking the perfect symmetry if it helps.
    if best_points is not None and time.time() < deadline - 15:
        
        def objective_full(x):
            if time.time() > deadline - 2:
                raise TimeoutError()
            p = x.reshape(N, 2)
            return _neg_score(p)
            
        try:
            res = minimize(
                objective_full, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if res.fun < 0:
                new_pts = res.x.reshape(N, 2)
                s = _score(new_pts)
                if s > best_score:
                    best_score = s
                    best_points = new_pts
                    
        except TimeoutError:
            pass
        except Exception:
            pass

    # 3. If no points found (shouldn't happen), fallback to regular polygon
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points

# EVOLVE-BLOCK-START
# The code above is the combined solution.
# It uses DE for global search over symmetric ring structures (inspiration 1)
# and Nelder-Mead for local refinement in full space (current program).
# EVOLVE-BLOCK-END
