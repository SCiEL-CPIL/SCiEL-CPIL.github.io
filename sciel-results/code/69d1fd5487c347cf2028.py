"""
Heilbronn convex regions n=14
Strategy: Adaptive arc-length reparameterization on a unit circle.
Instead of optimizing 14 independent angles (which forces the optimizer to manage both shape and spacing simultaneously in a coupled way), 
we optimize 13 positive "arc lengths" (fractions of the total 2*pi). 
The points are placed sequentially along the circle. This ensures the points are in convex position and naturally encourages the optimizer 
to find non-uniform spacing (clustering or spreading) that maximizes the minimum triangle area.
"""

import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    Uses vectorized cross products for efficiency.
    """
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    # We can vectorize the inner loop partially
    for i in range(N - 2):
        for j in range(i + 1, N - 1):
            # Vectorized calculation for all k > j
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
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

def _points_from_arcs(arcs):
    """
    Convert arc-length fractions to points on the unit circle.
    arcs: array of N-1 positive values.
    The last arc is determined by the constraint that the sum of all arcs is 2*pi.
    Actually, to keep it simple and bounded for DE, we treat the N-1 arcs as fractions of the total circle.
    Let f_i be the fraction of the circle for the i-th arc. Sum(f_i) = 1.
    The points are at cumulative sums of these fractions.
    
    To ensure convexity and avoid degeneracy (points too close), we enforce a minimum fraction.
    """
    # arcs are the first N-1 fractions. The last fraction is 1 - sum(arcs).
    # To make it a valid probability vector, we need sum(arcs) < 1.
    # DE works on bounded domains. Let's map the optimization variables to (min_frac, 1 - (N-1)*min_frac).
    # Simpler: Optimize N-1 variables in (0, 1) and normalize? No, that changes the geometry.
    
    # Let's use the strategy: Optimize N-1 variables x_i in [0, 1].
    # Let f_i = x_i / sum(x) for i=0..N-2.
    # f_{N-1} = 1 - sum(f_i).
    # This ensures sum f = 1 and all f_i >= 0.
    # However, if sum(x) is small, f_i can be large.
    # Better: Directly optimize N-1 variables representing the gaps.
    
    # Let's stick to the prompt's suggestion: "cumulative arc-length fractions".
    # Let the variables be the gaps between points.
    # g_0, ..., g_{N-2} are the variables.
    # g_{N-1} is the remaining gap.
    # We need g_i > 0 for all i.
    
    # To handle the constraint sum(g_i) = 2*pi, we can optimize N-1 variables and derive the last.
    # But DE requires bounds.
    # Let's optimize N-1 variables in [0, 2*pi].
    # Normalize them to sum to 2*pi? No, that's not linear.
    
    # Alternative: Optimize N-1 variables in [0, 1].
    # Let these be the weights w_i.
    # Let W = sum(w_i) + 1. (The +1 ensures the last gap is positive if w_i are non-negative? No.)
    
    # Let's use the standard "Dirichlet-like" parameterization for positive sums.
    # Let x_i be the optimization variables for i=0..N-2.
    # Let g_i = x_i for i=0..N-2.
    # Let g_{N-1} = max(0, 2*pi - sum(g_i)).
    # This is non-smooth at the boundary.
    
    # Better approach for DE:
    # Optimize N-1 variables y_i in [0, 1].
    # Map to gaps:
    # Let S = sum(y_i) + 1.
    # Let g_i = 2*pi * (y_i / S) for i=0..N-2.
    # Let g_{N-1} = 2*pi * (1 / S).
    # Sum(g_i) = 2*pi * (sum(y_i) + 1) / S = 2*pi.
    # All g_i > 0.
    # This is a smooth bijection from [0,1]^{N-1} to the simplex of positive gaps summing to 2*pi.
    
    y = arcs
    S = np.sum(y) + 1.0
    g = 2 * np.pi * y / S
    g_last = 2 * np.pi * (1.0 / S)
    gaps = np.concatenate([g, [g_last]])
    
    # Cumulative sums to get angles
    angles = np.cumsum(gaps)
    # The last angle is 2*pi, which is the same as 0. We only need N points, so we take the first N-1 cumulative sums?
    # No, we have N points. The gaps are between p0-p1, p1-p2, ..., p_{N-1}-p0.
    # There are N gaps.
    # My construction has N gaps: g_0..g_{N-2} (from y) and g_{N-1} (derived).
    # Total N gaps.
    # Points are at cumulative sums of the first N-1 gaps?
    # p_0 at 0.
    # p_1 at g_0.
    # ...
    # p_{N-1} at sum(g_0..g_{N-2}).
    # p_N (which is p_0) at sum(g_0..g_{N-1}) = 2*pi.
    
    # So we need N points.
    # angles[0] = 0
    # angles[1] = g_0
    # ...
    # angles[N-1] = sum(g_0..g_{N-2})
    
    # The `angles` array from cumsum(gaps) has length N.
    # angles[0] = g_0.
    # Wait, cumsum(gaps) gives [g_0, g_0+g_1, ..., sum(gaps)].
    # We want points at 0, g_0, g_0+g_1, ...
    # So we should do:
    # angles = np.concatenate([[0], np.cumsum(gaps[:-1])])
    
    angles = np.concatenate([[0], np.cumsum(gaps[:-1])])
    
    x = np.cos(angles)
    y_coords = np.sin(angles)
    return np.column_stack([x, y_coords])

def _angle_objective(y_flat):
    """
    Objective function for the arc-length parameterization.
    y_flat: N-1 variables in [0, 1].
    """
    points = _points_from_arcs(y_flat)
    score = _score(points)
    
    if score < 1e-6:
        return 1e10
        
    return -score

def construct():
    """
    Generate 14 points to maximize the minimum triangle area over convex hull area.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial guess: Regular polygon
    # For regular polygon, all gaps are equal: 2*pi/N.
    # In our parameterization:
    # g_i = 2*pi * (y_i / S) for i=0..N-2
    # g_{N-1} = 2*pi * (1 / S)
    # We want g_i = 2*pi/N for all i.
    # So 1/S = 1/N => S = N.
    # And y_i / S = 1/N => y_i = S/N = 1.
    # So initial y = np.ones(N-1).
    
    initial_y = np.ones(N - 1)
    
    best_y = initial_y.copy()
    best_score_val = _score(_points_from_arcs(initial_y))
    
    # Bounds for y: [0, 1]
    # Actually, if y_i can be 0, the gap is 0, which is degenerate.
    # But the mapping is continuous. DE will explore.
    # To avoid exact 0, we can set bounds to [1e-6, 1].
    bounds = [(1e-6, 1.0)] * (N - 1)
    
    # Run Differential Evolution
    try:
        res = differential_evolution(
            _angle_objective, 
            bounds,
            seed=42,
            maxiter=150,
            popsize=20,
            tol=1e-6,
            mutation=(0.5, 1.5),
            recombination=0.7,
            polish=False,
            init='latinhypercube'
        )
        
        if res.fun < 1e9:
            current_score = -res.fun
            if current_score > best_score_val:
                best_score_val = current_score
                best_y = res.x
    except Exception:
        pass
    
    # Polish with Nelder-Mead
    if time.time() < deadline - 20:
        try:
            res_nm = minimize(
                _angle_objective, 
                best_y, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res_nm.fun < _angle_objective(best_y):
                best_y = res_nm.x
                best_score_val = -res_nm.fun
        except Exception:
            pass
            
    # Final Polish with Powell
    if time.time() < deadline - 10:
        try:
            res_powell = minimize(
                _angle_objective, 
                best_y, 
                method='Powell', 
                options={'maxiter': 200, 'xtol': 1e-9, 'ftol': 1e-12}
            )
            if res_powell.fun < _angle_objective(best_y):
                best_y = res_powell.x
                best_score_val = -res_powell.fun
        except Exception:
            pass

    # Convert best y to points
    final_points = _points_from_arcs(best_y)
    
    # Normalize to unit area
    try:
        hull = ConvexHull(final_points)
        hull_area = hull.volume
    except Exception:
        # Fallback to regular polygon
        final_points = _points_from_arcs(initial_y)
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
    else:
        final_points = _points_from_arcs(initial_y)
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale

    return final_points
