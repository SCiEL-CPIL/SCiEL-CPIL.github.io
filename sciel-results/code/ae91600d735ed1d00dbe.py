import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    # Using numpy for speed if possible, but loops are fine for N=14
    # 14C3 = 364 triangles.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
        
    min_area = _min_triangle_area(points)
    
    return min_area / hull_area

def _neg_objective(points_flat):
    """
    Objective function for minimization.
    Normalizes points to unit hull area internally to enforce constraint,
    then returns negative score.
    """
    points = points_flat.reshape(-1, 2)
    
    # 1. Center
    points = points - np.mean(points, axis=0)
    
    # 2. Normalize to unit hull area
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6 # Penalty for degenerate hull
    
    scale = 1.0 / np.sqrt(hull_area)
    points = points * scale
    
    # 3. Calculate score (which is now just min_area since hull_area is 1)
    min_area = _min_triangle_area(points)
    
    # 4. Return negative for minimization
    return -min_area

def _smooth_neg_min_area(points_flat, tau=100.0):
    """
    Smooth surrogate for the negative min triangle area.
    Uses log-sum-exp approximation.
    """
    points = points_flat.reshape(-1, 2)
    
    # 1. Center
    points = points - np.mean(points, axis=0)
    
    # 2. Normalize to unit hull area
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e6
    
    scale = 1.0 / np.sqrt(hull_area)
    points = points * scale
    
    # 3. Calculate areas
    n = len(points)
    areas = np.zeros(364) # 14C3 = 364
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas[idx] = area
                idx += 1
    
    # 4. Smooth min: -1/tau * log(sum(exp(-tau * areas)))
    # To avoid overflow, we can use the log-sum-exp trick:
    # min(areas) is approximated by -1/tau * log(sum(exp(-tau * (areas - min_area)))) + min_area
    # But we want to minimize -min_area.
    # So we maximize min_area.
    # Equivalent to minimizing -min_area.
    # Smooth: - ( -1/tau * log(sum(exp(-tau * areas))) ) = 1/tau * log(sum(exp(-tau * areas)))
    # Wait.
    # f(x) = min(a_i).
    # Smooth approx: f_smooth(x) = -1/tau * log(sum(exp(-tau * a_i))).
    # We want to minimize -f(x) = -min(a_i).
    # So we minimize -f_smooth(x) = 1/tau * log(sum(exp(-tau * a_i))).
    
    # Let's verify.
    # If a = [1, 10], tau=10.
    # exp(-10*1) = exp(-10) ~ 4.5e-5.
    # exp(-10*10) = exp(-100) ~ 3.7e-44.
    # sum ~ 4.5e-5.
    # log(sum) ~ -10.
    # 1/tau * log(sum) = 0.1 * -10 = -1.
    # -min(a) = -1.
    # Matches.
    
    # Numerical stability:
    # log(sum(exp(-tau * a_i)))
    # Let m = min(a_i).
    # sum(exp(-tau * a_i)) = exp(-tau * m) * sum(exp(-tau * (a_i - m)))
    # log(sum) = -tau * m + log(sum(exp(-tau * (a_i - m))))
    # 1/tau * log(sum) = -m + 1/tau * log(sum(exp(-tau * (a_i - m))))
    # This is stable because a_i - m >= 0, so exp(-tau * (a_i - m)) <= 1.
    
    m = np.min(areas)
    shifted = areas - m
    sum_exp = np.sum(np.exp(-tau * shifted))
    smooth_val = -m + (1.0 / tau) * np.log(sum_exp)
    
    return smooth_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    best_score = -1.0
    best_points = None
    
    # Generate initial configurations
    configs = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(x0_reg)
    
    # 2. Regular 14-gon with noise
    configs.append(x0_reg + 0.05 * np.random.randn(N, 2))
    
    # 3. Two rings: 7 outer, 7 inner
    ang1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer = np.column_stack([np.cos(ang1), np.sin(ang1)])
    ang2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi/7
    inner = 0.5 * np.column_stack([np.cos(ang2), np.sin(ang2)])
    configs.append(np.vstack([outer, inner]))
    
    # 4. Three rings: 5, 5, 4
    ang1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    ring1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
    ang2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi/5
    ring2 = 0.6 * np.column_stack([np.cos(ang2), np.sin(ang2)])
    ang3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
    ring3 = 0.3 * np.column_stack([np.cos(ang3), np.sin(ang3)])
    configs.append(np.vstack([ring1, ring2, ring3]))
    
    # 5. Random points
    configs.append(np.random.randn(N, 2))
    
    # Add more diverse random starts
    for _ in range(5):
        configs.append(np.random.randn(N, 2))

    candidates = []

    for x_init in configs:
        if time.time() > deadline:
            break
            
        try:
            # Phase 1: Smooth optimization with Powell
            # Powell is robust and doesn't need gradients
            res_smooth = minimize(lambda x: _smooth_neg_min_area(x, tau=100.0), 
                                  x_init.flatten(), 
                                  method='Powell', 
                                  options={'maxiter': 500, 'xtol': 1e-6, 'ftol': 1e-8, 'adaptive': True})
            
            if time.time() > deadline:
                break
                
            p_opt = res_smooth.x.reshape(-1, 2)
            
            # Normalize p_opt for consistency
            p_opt = p_opt - np.mean(p_opt, axis=0)
            hull_area = _hull_area(p_opt)
            if hull_area > 1e-10:
                p_opt = p_opt / np.sqrt(hull_area)
            
            # Phase 2: Polish with exact objective using Nelder-Mead
            # Use tight tolerances for final precision
            res_exact = minimize(_neg_objective, 
                                 p_opt.flatten(), 
                                 method='Nelder-Mead', 
                                 options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if time.time() > deadline:
                break
                
            p_polished = res_exact.x.reshape(-1, 2)
            
            # Normalize final result
            p_polished = p_polished - np.mean(p_polished, axis=0)
            hull_area = _hull_area(p_polished)
            if hull_area > 1e-10:
                p_polished = p_polished / np.sqrt(hull_area)
            
            s_opt = _score(p_polished)
            
            if s_opt > best_score:
                best_score = s_opt
                best_points = p_polished
                
        except Exception as e:
            pass

    if best_points is None:
        # Fallback to regular polygon
        best_points = x0_reg
        best_points = best_points - np.mean(best_points, axis=0)
        hull_area = _hull_area(best_points)
        if hull_area > 1e-10:
             best_points = best_points / np.sqrt(hull_area)

    return best_points
