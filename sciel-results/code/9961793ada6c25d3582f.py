"""
Strategy: Centrally symmetric polygon optimization with global search.
For even N=14, optimal Heilbronn configurations are centrally symmetric.
We parameterize the 14 points as 7 antipodal pairs.
Each pair i is defined by a vector (x_i, y_i).
The 14 points are (x_i, y_i) and (-x_i, -y_i) for i=0..6.
To ensure a convex, non-degenerate hull, we can fix the angular order or simply rely on the optimizer to find a valid convex configuration.
A robust approach is to parameterize the 7 points in the first quadrant (or half-plane) with increasing angles, ensuring the hull is convex and contains the origin.
Actually, a simpler parameterization for centrally symmetric convex polygons:
Let the 7 vertices in the upper half-plane (including the x-axis) be v_0, ..., v_6.
v_0 = (r_0, 0)
v_k = (r_k cos(theta_k), r_k sin(theta_k)) for k=1..6
where 0 < theta_1 < ... < theta_6 < pi.
The other 7 vertices are -v_0, ..., -v_6.
This ensures the polygon is convex if the angles are increasing and radii are positive, and the polygon is centrally symmetric.
Total parameters: 7 radii + 6 angles = 13 parameters.
This is a 13-dimensional search space, much smaller than 28.
We use Differential Evolution for global search to avoid local minima, followed by Nelder-Mead for refinement.
"""

import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    """
    n = len(points)
    if n < 3:
        return float('inf')
    
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    # Vectorized computation of all triangle areas
    # i < j < k
    # Area = 0.5 * |det(P_j - P_i, P_k - P_i)|
    
    for i in range(n - 2):
        # P_i is points[i]
        # P_j - P_i for j > i
        # P_k - P_i for k > j
        
        # Efficient loop for small N
        dx_j = x[i+1:] - x[i]
        dy_j = y[i+1:] - y[i]
        
        for j_idx in range(len(dx_j)):
            j = i + 1 + j_idx
            dx_ij = dx_j[j_idx]
            dy_ij = dy_j[j_idx]
            
            if j + 1 >= n:
                continue
            
            dx_ik = x[j+1:] - x[i]
            dy_ik = y[j+1:] - y[i]
            
            cross = dx_ij * dy_ik - dx_ik * dy_ij
            areas = 0.5 * np.abs(cross)
            
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
                    if min_area < 1e-10:
                        return min_area
                        
    return min_area

def _score(points):
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

def _points_from_sym_params(params):
    """
    Convert parameters to 14 points.
    params: array of shape (13,)
    params[0:7] are radii r_0..r_6
    params[7:13] are angles theta_1..theta_6 (in (0, pi))
    """
    r = params[:7]
    theta = params[7:13]
    
    # Ensure positivity and ordering
    # r > 0
    # 0 < theta_1 < ... < theta_6 < pi
    
    # Soft constraints via transformation
    # r_i = exp(r_param_i)
    # theta_i = pi * sigmoid(theta_param_i) ... but we need ordering.
    
    # Better: parameterize angles as cumulative sums of positive values.
    # Let t_1..t_6 be positive.
    # theta_k = sum_{j=1}^k t_j
    # We need theta_6 < pi.
    # So sum_{j=1}^6 t_j < pi.
    # Let t_j = pi * u_j / sum(u) where u_j > 0.
    # Then theta_k = pi * (sum_{j=1}^k u_j) / (sum_{j=1}^6 u_j)
    # This ensures 0 < theta_1 < ... < theta_6 < pi.
    
    # Radii: r_i = exp(r_param_i)
    
    r_exp = np.exp(r)
    
    # Angles from raw params
    u = np.exp(theta) # theta here are raw params for angles, let's call them a
    # Wait, params[7:13] are the raw values for angles.
    # Let's map them to u > 0.
    u = np.exp(theta)
    s = np.sum(u)
    if s < 1e-10:
        s = 1e-10
    theta_norm = np.cumsum(u) / s * np.pi
    
    # Construct points
    # v_0 = (r_0, 0)
    # v_k = (r_k cos(theta_k), r_k sin(theta_k)) for k=1..6
    
    x = np.zeros(7)
    y = np.zeros(7)
    
    x[0] = r_exp[0]
    y[0] = 0.0
    
    for k in range(1, 7):
        x[k] = r_exp[k] * np.cos(theta_norm[k-1])
        y[k] = r_exp[k] * np.sin(theta_norm[k-1])
        
    v_upper = np.column_stack([x, y])
    
    # Full 14 points
    v_lower = -v_upper
    
    points = np.vstack([v_upper, v_lower])
    
    return points

def _objective(params):
    """
    Objective function to minimize (negative score).
    """
    points = _points_from_sym_params(params)
    
    # Check if points are distinct and form a valid convex hull
    # If hull area is 0, return large value
    score = _score(points)
    
    if score < 1e-6:
        return 1e10
        
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Bounds for parameters
    # Radii: exp(r) should be reasonable. r in [-5, 5] -> r_exp in [0.006, 148]
    # Angles: exp(a) should be reasonable. a in [-5, 5]
    
    bounds = [(-5, 5)] * 13
    
    # Initial guess: Regular 14-gon
    # Regular 14-gon is centrally symmetric.
    # Vertices at angles 0, 2pi/14, ..., 12pi/14
    # Upper half: 0, 2pi/14, 4pi/14, 6pi/14, 8pi/14, 10pi/14, 12pi/14
    # Wait, 14 points. 7 in upper, 7 in lower.
    # Angles for upper: 0, 2pi/14, 4pi/14, 6pi/14, 8pi/14, 10pi/14, 12pi/14
    # But 12pi/14 = 6pi/7 < pi.
    # So theta_1..theta_6 correspond to 2pi/14, 4pi/14, ..., 12pi/14.
    
    r_init = np.zeros(7) # exp(0) = 1
    # We need theta_norm to be [2pi/14, 4pi/14, ..., 12pi/14]
    # theta_norm_k = pi * (sum_{j=1}^k u_j) / S
    # We want theta_norm_k = 2pi k / 14 = pi k / 7
    # So (sum_{j=1}^k u_j) / S = k / 7
    # This implies u_1 = u_2 = ... = u_6.
    # So we can set u_j = 1 for all j.
    # u = exp(a) => a = 0.
    
    a_init = np.zeros(6)
    
    params_init = np.concatenate([r_init, a_init])
    
    best_score_val = -_objective(params_init)
    best_params = params_init.copy()
    
    # Differential Evolution for global search
    if time.time() < deadline - 30:
        try:
            res_de = differential_evolution(
                _objective,
                bounds=bounds,
                seed=42,
                maxiter=50,
                tol=1e-6,
                popsize=15,
                polish=False
            )
            if -res_de.fun > best_score_val:
                best_score_val = -res_de.fun
                best_params = res_de.x
        except Exception:
            pass
    
    # Nelder-Mead for refinement
    if time.time() < deadline - 10:
        try:
            res_nm = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if -res_nm.fun > best_score_val:
                best_score_val = -res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass
            
    # L-BFGS-B for further refinement
    if time.time() < deadline - 5:
        try:
            res_lbfgs = minimize(
                _objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if -res_lbfgs.fun > best_score_val:
                best_score_val = -res_lbfgs.fun
                best_params = res_lbfgs.x
        except Exception:
            pass
            
    # Reconstruct points
    points = _points_from_sym_params(best_params)
    
    # Normalize to unit area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
        return final_points
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = points * scale
    else:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale

    return final_points
