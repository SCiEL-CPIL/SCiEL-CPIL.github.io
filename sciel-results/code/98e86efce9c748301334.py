"""Optimized solution for Heilbronn convex regions - 13 points."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
from itertools import combinations

N = 13

def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 1.0

def _score(pts):
    """Score = min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a < 1e-10:
        return -1.0
    min_ta = _min_triangle_area(pts)
    return min_ta / hull_a

def _negative_score(pts):
    return -_score(pts)

def _make_regular_polygon(n, radius=1.0, center=(0,0), phase=0.0):
    """Create a regular n-gon."""
    angles = 2 * np.pi * np.arange(n) / n + phase
    x = radius * np.cos(angles) + center[0]
    y = radius * np.sin(angles) + center[1]
    return np.column_stack((x, y))

def _make_multi_shell(n_outer, n_inner, r_outer=1.0, r_inner=0.5, phase_outer=0.0, phase_inner=0.0):
    """Create multi-shell configuration: outer regular polygon + inner regular polygon."""
    outer = _make_regular_polygon(n_outer, r_outer, (0,0), phase_outer)
    inner = _make_regular_polygon(n_inner, r_inner, (0,0), phase_inner)
    return np.vstack([outer, inner])

def _smooth_min_triangle_area(pts, beta=100.0):
    """Compute smooth approximation of minimum triangle area using log-sum-exp."""
    n = len(pts)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                areas[i,j,k] = area
                areas[k,j,i] = area
                areas[j,k,i] = area
                areas[i,k,j] = area
                areas[k,i,j] = area
                areas[j,i,k] = area
    
    # Only consider unique triples (i < j < k)
    min_areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                min_areas.append(area)
    
    min_areas = np.array(min_areas)
    # Soft minimum via log-sum-exp
    log_min = -np.log(np.sum(np.exp(-beta * min_areas))) / beta
    return log_min

def _param_to_points(params, n_outer, n_inner):
    """Convert parameter vector to point coordinates.
    
    params: [r_inner, phase_outer, phase_inner]
    r_outer is fixed at 1.0
    """
    r_inner = params[0]
    phase_outer = params[1]
    phase_inner = params[2]
    
    outer = _make_regular_polygon(n_outer, 1.0, (0,0), phase_outer)
    inner = _make_regular_polygon(n_inner, r_inner, (0,0), phase_inner)
    return np.vstack([outer, inner])

def _param_to_points_smooth(params, n_outer, n_inner, beta=100.0):
    """Convert parameter vector to point coordinates (smooth version)."""
    return _param_to_points(params, n_outer, n_inner)

def _neg_score_param(params, n_outer, n_inner, beta=100.0):
    """Negative score for parameterized optimization with smooth objective."""
    pts = _param_to_points(params, n_outer, n_inner)
    hull_a = _hull_area(pts)
    if hull_a < 1e-10:
        return 1.0
    min_ta = _smooth_min_triangle_area(pts, beta)
    return -min_ta / hull_a

def construct():
    """Construct 13 points to maximize min triangle area / hull area."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_score = -np.inf
    best_pts = None
    
    # Initial configurations to try
    initial_configs = []
    
    # 1. Regular 13-gon
    initial_configs.append(_make_regular_polygon(13, 1.0, (0,0), 0.0))
    
    # 2. Multi-shell: 7 outer + 6 inner
    for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for phase in [0.0, np.pi/7, np.pi/6, np.pi/12]:
            initial_configs.append(_make_multi_shell(7, 6, 1.0, r_inner, 0.0, phase))
    
    # 3. Multi-shell: 6 outer + 7 inner
    for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for phase in [0.0, np.pi/6, np.pi/7, np.pi/12]:
            initial_configs.append(_make_multi_shell(6, 7, 1.0, r_inner, 0.0, phase))
    
    # 4. Multi-shell: 8 outer + 5 inner
    for r_inner in [0.3, 0.4, 0.5, 0.6]:
        for phase in [0.0, np.pi/8, np.pi/5]:
            initial_configs.append(_make_multi_shell(8, 5, 1.0, r_inner, 0.0, phase))
    
    # 5. Multi-shell: 5 outer + 8 inner
    for r_inner in [0.3, 0.4, 0.5, 0.6]:
        for phase in [0.0, np.pi/5, np.pi/8]:
            initial_configs.append(_make_multi_shell(5, 8, 1.0, r_inner, 0.0, phase))
    
    # 6. Multi-shell: 9 outer + 4 inner
    for r_inner in [0.3, 0.4, 0.5]:
        initial_configs.append(_make_multi_shell(9, 4, 1.0, r_inner, 0.0, 0.0))
    
    # 7. Multi-shell: 4 outer + 9 inner
    for r_inner in [0.3, 0.4, 0.5]:
        initial_configs.append(_make_multi_shell(4, 9, 1.0, r_inner, 0.0, 0.0))
    
    # 8. Regular 13-gon with different phases
    for phase in [np.pi/13, 2*np.pi/13]:
        initial_configs.append(_make_regular_polygon(13, 1.0, (0,0), phase))
    
    # Evaluate all initial configs
    for cfg in initial_configs:
        if time.time() > deadline:
            break
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_pts = cfg.copy()
    
    # Parameterized optimization over multi-shell configurations
    if time.time() < deadline - 20.0 and best_pts is not None:
        # Try different partitions and optimize parameters
        partitions = [(7, 6), (6, 7), (8, 5), (5, 8), (9, 4), (4, 9)]
        
        for n_outer, n_inner in partitions:
            if time.time() > deadline - 15.0:
                break
            
            # Multiple initial parameter guesses
            for r_init in [0.3, 0.4, 0.5, 0.6, 0.7]:
                for phase_o_init in [0.0, np.pi/n_outer]:
                    for phase_i_init in [0.0, np.pi/n_inner, np.pi/(2*n_inner)]:
                        if time.time() > deadline - 10.0:
                            break
                        
                        params_init = np.array([r_init, phase_o_init, phase_i_init])
                        
                        try:
                            # Use L-BFGS-B with smooth objective
                            result = minimize(
                                _neg_score_param,
                                params_init,
                                args=(n_outer, n_inner, 50.0),
                                method='L-BFGS-B',
                                bounds=[(0.1, 0.95), (-np.pi, np.pi), (-np.pi, np.pi)],
                                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
                            )
                            
                            if time.time() < deadline:
                                opt_params = result.x
                                opt_pts = _param_to_points(opt_params, n_outer, n_inner)
                                s = _score(opt_pts)
                                if s > best_score:
                                    best_score = s
                                    best_pts = opt_pts.copy()
                                    
                                    # Further polish with tighter tolerance
                                    if time.time() < deadline - 5.0:
                                        result2 = minimize(
                                            _neg_score_param,
                                            opt_params,
                                            args=(n_outer, n_inner, 100.0),
                                            method='L-BFGS-B',
                                            bounds=[(0.1, 0.95), (-np.pi, np.pi), (-np.pi, np.pi)],
                                            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                                        )
                                        if time.time() < deadline:
                                            opt_params2 = result2.x
                                            opt_pts2 = _param_to_points(opt_params2, n_outer, n_inner)
                                            s2 = _score(opt_pts2)
                                            if s2 > best_score:
                                                best_score = s2
                                                best_pts = opt_pts2.copy()
                        except Exception:
                            pass
    
    # Final full-coordinate optimization from best config
    if time.time() < deadline - 5.0 and best_pts is not None:
        x0 = best_pts.flatten()
        try:
            result = minimize(
                _negative_score,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if time.time() < deadline:
                opt_pts = result.x.reshape(N, 2)
                s = _score(opt_pts)
                if s > best_score:
                    best_score = s
                    best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Final: ensure we return something valid
    if best_pts is None:
        best_pts = _make_regular_polygon(13, 1.0, (0,0), 0.0)
    
    return best_pts
