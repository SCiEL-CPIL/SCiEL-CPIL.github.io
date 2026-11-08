import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 13

def _all_triangle_areas(points):
    """Compute all triangle areas vectorized."""
    n = len(points)
    # Get all triples
    indices = np.array(list(itertools.combinations(range(n), 3)))
    a = points[indices[:, 0]]
    b = points[indices[:, 1]]
    c = points[indices[:, 2]]
    # Cross product magnitude / 2
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    return 0.5 * np.abs(cross)

def _hull_area(points):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the ratio of minimum triangle area to convex hull area."""
    areas = _all_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _soft_min_objective(points, temperature=0.01):
    """
    Smooth surrogate for the minimum triangle area ratio using log-sum-exp.
    """
    areas = _all_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    
    neg_areas = -areas / temperature
    max_neg = np.max(neg_areas)
    soft_min = -temperature * (max_neg + np.log(np.sum(np.exp(neg_areas - max_neg))))
    
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    
    return soft_min / hull_area

def _negative_soft_min_ratio(points_flat, temperature=0.01):
    """Negative of soft-min ratio for minimization."""
    points = points_flat.reshape(N, 2)
    return -_soft_min_objective(points, temperature)

def _negative_exact_ratio(points_flat):
    """Negative of exact min ratio for minimization."""
    points = points_flat.reshape(N, 2)
    return -_score(points)

# EVOLVE-BLOCK-START
def _make_two_shell(r_inner, n_outer, n_inner, angle_offset=0.0):
    """Create points on two concentric regular polygons."""
    outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + angle_offset
    inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + angle_offset + np.pi / n_inner
    
    outer_pts = np.column_stack((r_inner * np.cos(outer_angles), r_inner * np.sin(outer_angles)))
    inner_pts = np.column_stack((np.cos(inner_angles), np.sin(inner_angles)))
    
    return np.vstack([outer_pts, inner_pts])

def _optimize_partition(r_inner, n_outer, n_inner, start_time, deadline):
    """Optimize a two-shell configuration."""
    points = _make_two_shell(r_inner, n_outer, n_inner)
    current_score = _score(points)
    
    # Optimize with soft-min surrogate
    temperatures = [0.05, 0.02, 0.01, 0.005]
    for temp in temperatures:
        if time.time() > deadline:
            break
        try:
            result = minimize(
                _negative_soft_min_ratio,
                points.flatten(),
                args=(temp,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            new_points = result.x.reshape(N, 2)
            new_score = _score(new_points)
            if new_score > current_score:
                points = new_points
                current_score = new_score
        except Exception:
            pass
    
    # Fine-tune with exact objective
    if time.time() < deadline:
        try:
            result = minimize(
                _negative_exact_ratio,
                points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            new_points = result.x.reshape(N, 2)
            new_score = _score(new_points)
            if new_score > current_score:
                points = new_points
                current_score = new_score
        except Exception:
            pass
    
    return points, current_score

def construct():
    """
    Constructs 13 points to maximize the minimum triangle area relative to convex hull area.
    Uses multiple initializations: regular polygon and two-shell configurations.
    """
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    best_points = None
    best_score = 0.0
    
    # Try multiple initializations
    initializations = [
        # Regular 13-gon
        lambda: np.column_stack((np.cos(np.linspace(0, 2*np.pi, N, endpoint=False)), 
                                 np.sin(np.linspace(0, 2*np.pi, N, endpoint=False)))),
        # Two-shell: 7 outer, 6 inner
        lambda: _make_two_shell(0.7, 7, 6),
        # Two-shell: 6 outer, 7 inner
        lambda: _make_two_shell(0.8, 6, 7),
        # Two-shell: 8 outer, 5 inner
        lambda: _make_two_shell(0.6, 8, 5),
        # Two-shell: 5 outer, 8 inner
        lambda: _make_two_shell(0.9, 5, 8),
    ]
    
    for init_func in initializations:
        if time.time() > deadline:
            break
        
        try:
            points, score = _optimize_partition(0.7, 7, 6, start_time, deadline)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
        except Exception:
            pass
    
    # If no valid result, fall back to regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_points
# EVOLVE-BLOCK-END
