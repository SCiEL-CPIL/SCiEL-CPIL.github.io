"""Convex-position parameterization for Heilbronn n=14: angles+radii enforce convex order."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _build_points(angles, radii):
    """
    Build points from angular order and radii.
    This ensures points are in convex position if radii are positive and angles are sorted.
    """
    cx = np.cos(angles) * radii
    cy = np.sin(angles) * radii
    return np.column_stack([cx, cy])

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    Vectorized inner loop for speed.
    """
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(N):
        xi, yi = x[i], y[i]
        dx = x[i+1:] - xi
        dy = y[i+1:] - yi
        
        # For each j > i, compute cross product with all k > j
        # dx[j-1], dy[j-1] is vector from i to j
        # dx[j:], dy[j:] are vectors from i to k
        
        for j in range(1, N - i):
            dxj, dyj = dx[j-1], dy[j-1]
            if j < N - i:
                # cross product of (dxj, dyj) and (dx[j:], dy[j:])
                cross = dxj * dy[j:] - dx[j:] * dyj
                areas = 0.5 * np.abs(cross)
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
                    
    return min_area

def _score_points(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _neg_score_params(params):
    """
    Objective function for minimization.
    params: (2N,) array, first N are angles, next N are radii.
    We maximize score, so minimize -score.
    """
    angles = params[:N]
    radii = params[N:]
    
    # Ensure radii are positive to maintain convex position from origin
    # If any radius is <= 0, it's a degenerate case
    if np.any(radii <= 0):
        return 1e10
        
    points = _build_points(angles, radii)
    
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    score = _score_points(points)
    
    if score < 1e-6:
        return 1e10
        
    return -score

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    
    Strategy:
    1. Parameterize points by (angle, radius) to enforce convex position.
    2. Initialize with regular 14-gon (equal angles, equal radii).
    3. Optimize in the 28D (angle, radius) space using Nelder-Mead.
    4. This reduces the effective degrees of freedom for the hull shape
       and avoids interior points which are suboptimal for maximizing min triangle area.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initialization: Regular 14-gon
    # Angles are equally spaced
    angles0 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Radii are equal (unit circle initially, scaling doesn't affect score)
    radii0 = np.ones(N)
    
    params0 = np.concatenate([angles0, radii0])
    
    best_params = params0.copy()
    best_points = _build_points(angles0, radii0)
    best_score_val = _score_points(best_points)
    
    # 2. Nelder-Mead Refinement
    # We optimize over angles and radii.
    # Note: Angles are cyclic, but NM works on R^N. 
    # To handle periodicity, we could unwrap, but for small deformations from regular polygon,
    # standard NM should work well.
    
    try:
        res_nm = minimize(
            _neg_score_params, 
            best_params, 
            method='Nelder-Mead', 
            options={
                'maxiter': 5000, 
                'xatol': 1e-8, 
                'fatol': 1e-10, 
                'adaptive': True
            }
        )
        
        if res_nm.fun < -best_score_val: # i.e. -score > -best_score
            best_score_val = -res_nm.fun
            best_params = res_nm.x
            best_points = _build_points(best_params[:N], best_params[N:])
    except Exception:
        pass
    
    # 3. Second round of Nelder-Mead with tighter tolerances or different start if time permits
    if time.time() < deadline - 20:
        try:
            # Perturb slightly to escape potential local minima
            # Or just run again with tighter settings
            res_nm2 = minimize(
                _neg_score_params, 
                best_params, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 5000, 
                    'xatol': 1e-10, 
                    'fatol': 1e-12, 
                    'adaptive': True
                }
            )
            if res_nm2.fun < -best_score_val:
                best_score_val = -res_nm2.fun
                best_params = res_nm2.x
                best_points = _build_points(best_params[:N], best_params[N:])
        except Exception:
            pass
            
    # 4. Powell's Method Polish (often good for finishing off NM)
    if time.time() < deadline - 10:
        try:
            res_powell = minimize(
                _neg_score_params, 
                best_params, 
                method='Powell', 
                options={
                    'maxiter': 2000, 
                    'xtol': 1e-10, 
                    'ftol': 1e-12
                }
            )
            if res_powell.fun < -best_score_val:
                best_score_val = -res_powell.fun
                best_params = res_powell.x
                best_points = _build_points(best_params[:N], best_params[N:])
        except Exception:
            pass

    # 5. Final Verification and Normalization
    final_points = best_points
    
    # Check validity
    try:
        hull = ConvexHull(final_points)
        hull_area = hull.volume
    except Exception:
        # Fallback to regular polygon if hull fails
        final_points = _build_points(angles0, radii0)
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
    else:
        # Degenerate case, return regular polygon normalized
        final_points = _build_points(angles0, radii0)
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale

    return final_points
# EVOLVE-BLOCK-END
