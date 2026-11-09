import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13

def _min_triangle_area_and_indices(points):
    """Return (min_area, (i,j,k)) for the minimum-area triangle."""
    n = len(points)
    min_area = np.inf
    min_idx = (0, 1, 2)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
                if area < min_area:
                    min_area = area
                    min_idx = (i, j, k)
    return min_area, min_idx

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def get_score(points):
    points = np.asarray(points, dtype=float)
    if points.shape != (N, 2):
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area, _ = _min_triangle_area_and_indices(points)
    return min_area / hull_area

def _objective(x, deadline):
    """Objective to minimize: -score (i.e., negative of min_area/hull_area)."""
    points = x.reshape(-1, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    min_area, _ = _min_triangle_area_and_indices(points)
    return -min_area / hull_area

def _optimize_from(points, deadline, max_iter=200):
    """
    Use scipy.optimize.minimize with Nelder-Mead to maximize min triangle area / hull area.
    """
    points = points.copy()
    best_points = points.copy()
    best_score = get_score(points)
    
    # Try Nelder-Mead
    try:
        x0 = points.flatten()
        result = minimize(
            _objective,
            x0,
            args=(deadline,),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        if result.success or result.fun < 1e10:
            opt_points = result.x.reshape(-1, 2)
            opt_score = get_score(opt_points)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points
    except Exception:
        pass
    
    return best_points, best_score

def _generate_regular_polygon_configs():
    """Generate structured initial configurations."""
    configs = []
    
    # Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Regular 12-gon + center
    angles12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts12 = np.column_stack([np.cos(angles12), np.sin(angles12)])
    configs.append(np.vstack([pts12, np.array([[0.0, 0.0]])]))
    
    # Regular 11-gon + 2 inner
    angles11 = np.linspace(0, 2*np.pi, 11, endpoint=False)
    pts11 = np.column_stack([np.cos(angles11), np.sin(angles11)])
    configs.append(np.vstack([pts11, np.array([[0.5, 0.0], [-0.5, 0.0]])]))
    
    # Regular 10-gon + 3 inner (triangle at center)
    angles10 = np.linspace(0, 2*np.pi, 10, endpoint=False)
    pts10 = np.column_stack([np.cos(angles10), np.sin(angles10)])
    inner_angles = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
    inner = np.column_stack([0.4*np.cos(inner_angles), 0.4*np.sin(inner_angles)])
    configs.append(np.vstack([pts10, inner]))
    
    # Regular 9-gon + 4 inner (square at center)
    angles9 = np.linspace(0, 2*np.pi, 9, endpoint=False)
    pts9 = np.column_stack([np.cos(angles9), np.sin(angles9)])
    inner_angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
    inner = np.column_stack([0.35*np.cos(inner_angles), 0.35*np.sin(inner_angles)])
    configs.append(np.vstack([pts9, inner]))
    
    # Two concentric rings: 7 outer, 6 inner
    angles7 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer = np.column_stack([np.cos(angles7), np.sin(angles7)])
    angles6 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    inner = np.column_stack([0.5*np.cos(angles6), 0.5*np.sin(angles6)])
    configs.append(np.vstack([outer, inner]))
    
    # 8 outer, 5 inner
    angles8 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    outer = np.column_stack([np.cos(angles8), np.sin(angles8)])
    angles5 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    inner = np.column_stack([0.45*np.cos(angles5), 0.45*np.sin(angles5)])
    configs.append(np.vstack([outer, inner]))
    
    # 6 outer, 7 inner
    angles6 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    outer = np.column_stack([np.cos(angles6), np.sin(angles6)])
    angles7 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    inner = np.column_stack([0.6*np.cos(angles7), 0.6*np.sin(angles7)])
    configs.append(np.vstack([outer, inner]))
    
    # 5 outer, 8 inner
    angles5 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    outer = np.column_stack([np.cos(angles5), np.sin(angles5)])
    angles8 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    inner = np.column_stack([0.55*np.cos(angles8), 0.55*np.sin(angles8)])
    configs.append(np.vstack([outer, inner]))
    
    # 4 outer, 9 inner
    angles4 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    outer = np.column_stack([np.cos(angles4), np.sin(angles4)])
    angles9 = np.linspace(0, 2*np.pi, 9, endpoint=False)
    inner = np.column_stack([0.5*np.cos(angles9), 0.5*np.sin(angles9)])
    configs.append(np.vstack([outer, inner]))
    
    # Random with various seeds
    for seed in [1, 7, 13, 42, 99, 123, 456, 789]:
        rng = np.random.default_rng(seed)
        pts = rng.random((N, 2))
        configs.append(pts)
    
    # More structured random: points on two circles
    for r_inner in [0.3, 0.4, 0.5, 0.6]:
        for n_outer in [5, 6, 7, 8]:
            n_inner = N - n_outer
            angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
            outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
            angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + np.pi/n_outer
            inner = np.column_stack([r_inner*np.cos(angles_inner), r_inner*np.sin(angles_inner)])
            configs.append(np.vstack([outer, inner]))
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    configs = _generate_regular_polygon_configs()
    
    best_score = -np.inf
    best_points = None
    
    for idx, init_points in enumerate(configs):
        if time.time() > deadline - 10.0:
            break
        
        try:
            optimized, score = _optimize_from(init_points, deadline - 10.0)
            if score > best_score:
                best_score = score
                best_points = optimized
        except Exception:
            pass
    
    if best_points is None:
        best_points = configs[0]
    
    return best_points
