"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _all_triangle_areas(points):
    """Compute all triangle areas for N points, vectorized."""
    n = len(points)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                a = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas[i, j, k] = a
                areas[j, i, k] = a
                areas[k, j, i] = a
    return areas

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: np.array of shape (N, 2)
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    areas = _all_triangle_areas(points)
    min_area = areas.min()
    
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def _soft_min_score(points, temperature=0.01):
    """
    Smooth surrogate: soft-min of triangle areas / hull area.
    Uses log-sum-exp for the soft-min.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    areas = _all_triangle_areas(points)
    # Only consider valid triangles (i < j < k)
    valid = np.zeros_like(areas, dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                valid[i, j, k] = True
    
    a = areas[valid]
    if len(a) == 0:
        return 0.0
    
    # Soft-min: -log(sum(exp(-a/T))) * T
    # For numerical stability
    a_shifted = a - a.max()
    log_sum = np.log(np.sum(np.exp(a_shifted / temperature)))
    soft_min = (a.max() - temperature * log_sum)
    
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
        
    return soft_min / hull_area

def _neg_soft_score(points_flat, temperature=0.01):
    """Negative soft score for minimization with L-BFGS-B."""
    points = points_flat.reshape(-1, 2)
    return -_soft_min_score(points, temperature)

def _neg_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _make_ring_config(n_outer, n_inner, inner_radius, phase_shift=0.0):
    """Create a two-ring configuration."""
    outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + phase_shift
    
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner = np.column_stack([inner_radius * np.cos(inner_angles), 
                             inner_radius * np.sin(inner_angles)])
    return np.vstack([outer, inner])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_score = _score(x0)
    best_points = x0.copy()
    
    # Generate diverse initial configurations
    configs = [
        x0.copy(),
        x0 + 0.05 * np.random.randn(N, 2),
        x0 + 0.1 * np.random.randn(N, 2),
        _make_ring_config(7, 7, 0.5, np.pi/7),
        _make_ring_config(7, 7, 0.6, np.pi/7),
        _make_ring_config(7, 7, 0.7, np.pi/7),
        _make_ring_config(6, 8, 0.5, 0.0),
        _make_ring_config(8, 6, 0.5, 0.0),
        _make_ring_config(5, 9, 0.6, np.pi/5),
        _make_ring_config(9, 5, 0.6, np.pi/9),
    ]
    
    # Add some jittered ring configs
    for r in [0.4, 0.55, 0.65, 0.75]:
        cfg = _make_ring_config(7, 7, r, np.pi/7)
        cfg = cfg + 0.03 * np.random.randn(N, 2)
        configs.append(cfg)
    
    # Phase 1: Use L-BFGS-B with smooth surrogate for fast convergence
    for idx, x_init in enumerate(configs):
        if time.time() > deadline - 15.0:
            break
            
        try:
            # Smooth optimization with L-BFGS-B
            res_smooth = minimize(_neg_soft_score, x_init.flatten(), 
                                  args=(0.05,),
                                  method='L-BFGS-B', 
                                  options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5})
            
            if time.time() > deadline - 10.0:
                break
                
            p_smooth = res_smooth.x.reshape(-1, 2)
            s_smooth = _score(p_smooth)
            
            # Phase 2: Polish with Nelder-Mead on exact objective
            if s_smooth > best_score * 0.9:  # Only polish if promising
                res_nm = minimize(_neg_score, p_smooth.flatten(), 
                                  method='Nelder-Mead',
                                  options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-7})
                
                if time.time() > deadline - 5.0:
                    break
                    
                p_opt = res_nm.x.reshape(-1, 2)
                s_opt = _score(p_opt)
                
                if s_opt > best_score:
                    best_score = s_opt
                    best_points = p_opt
        except Exception:
            pass
    
    # Phase 3: Final polish on best point found
    if time.time() < deadline - 3.0:
        try:
            res_final = minimize(_neg_score, best_points.flatten(),
                                 method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
            p_final = res_final.x.reshape(-1, 2)
            s_final = _score(p_final)
            if s_final > best_score:
                best_score = s_final
                best_points = p_final
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
