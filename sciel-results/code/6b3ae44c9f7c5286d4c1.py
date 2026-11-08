N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

def _min_triangle_area(points):
    """Compute minimum triangle area among all triples."""
    n = len(points)
    min_area = float('inf')
    for i, j, k in combinations(range(n), 3):
        p1, p2, p3 = points[i], points[j], points[k]
        area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
        if area < min_area:
            min_area = area
    return min_area

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / hull area. Higher is better."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _neg_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(N, 2)
    return -_score(points)

def _regular_polygon(n, radius=1.0, center=(0,0)):
    """Generate regular n-gon centered at center with given radius."""
    angles = 2 * np.pi * np.arange(n) / n
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])

def _initial_points_regular():
    """Regular 13-gon."""
    return _regular_polygon(N, radius=1.0, center=(0,0))

def _initial_points_mixed():
    """Regular 12-gon on boundary + 1 center point, then adjust."""
    pts = _regular_polygon(12, radius=1.0, center=(0,0))
    center_pt = np.array([[0.0, 0.0]])
    pts = np.vstack([pts, center_pt])
    idx = np.random.RandomState(42).permutation(N)
    return pts[idx]

def _soft_min_ratio(points, temperature=0.01):
    """
    Smooth surrogate for the minimum triangle area ratio using log-sum-exp.
    This makes the objective differentiable for gradient-based optimization.
    """
    areas = []
    for i, j, k in combinations(range(N), 3):
        p1, p2, p3 = points[i], points[j], points[k]
        area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
        areas.append(area)
    areas = np.array(areas)
    if len(areas) == 0:
        return 0.0
    
    # Use soft-min: -log-sum-exp(-areas/temperature) * temperature
    neg_areas = -areas / temperature
    max_neg = np.max(neg_areas)
    soft_min = -temperature * (max_neg + np.log(np.sum(np.exp(neg_areas - max_neg))))
    
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    
    return soft_min / hull_area

def _neg_soft_ratio(points_flat, temperature=0.01):
    """Negative of soft-min ratio for minimization."""
    points = points_flat.reshape(N, 2)
    return -_soft_min_ratio(points, temperature)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_score = -1.0
    best_points = None
    
    # Initial configurations
    initial_configs = [
        _initial_points_regular(),
        _initial_points_mixed(),
    ]
    
    # Also try a few random perturbations of regular polygon
    rng = np.random.RandomState(42)
    for _ in range(3):
        base = _regular_polygon(N, radius=1.0, center=(0,0))
        perturbed = base + rng.normal(0, 0.1, size=(N, 2))
        initial_configs.append(perturbed)
    
    for init_pts in initial_configs:
        if time.time() > deadline:
            break
        
        # Evaluate initial score
        init_score = _score(init_pts)
        if init_score > best_score:
            best_score = init_score
            best_points = init_pts.copy()
        
        # Try Nelder-Mead optimization on exact objective
        try:
            result = minimize(
                _neg_score,
                init_pts.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            opt_pts = result.x.reshape(N, 2)
            opt_score = _score(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # If we still have time, do more random restarts with exact optimization
    if time.time() < deadline - 20:
        for restart in range(5):
            if time.time() > deadline:
                break
            
            # Random perturbation of best
            if best_points is not None:
                init_pts = best_points + rng.normal(0, 0.05, size=(N, 2))
            else:
                init_pts = _initial_points_regular()
            
            try:
                result = minimize(
                    _neg_score,
                    init_pts.flatten(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                
                opt_pts = result.x.reshape(N, 2)
                opt_score = _score(opt_pts)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
            except Exception:
                pass
    
    # NEW: Use soft-min surrogate for gradient-based refinement (L-BFGS-B)
    # This can escape local minima that Nelder-Mead gets stuck in
    if time.time() < deadline - 10 and best_points is not None:
        try:
            # Start from the best points found so far
            x0 = best_points.flatten()
            
            # Try L-BFGS-B with the soft-min objective
            result = minimize(
                _neg_soft_ratio,
                x0,
                method='L-BFGS-B',
                args=(0.01,),  # temperature
                options={
                    'maxiter': 500,
                    'ftol': 1e-10,
                    'gtol': 1e-8
                }
            )
            
            opt_pts = result.x.reshape(N, 2)
            opt_score = _score(opt_pts)  # Use exact score for comparison
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass
        
        # Try with different temperatures
        if time.time() < deadline - 5 and best_points is not None:
            for temp in [0.005, 0.02, 0.05]:
                if time.time() > deadline - 5:
                    break
                try:
                    x0 = best_points.flatten()
                    result = minimize(
                        _neg_soft_ratio,
                        x0,
                        method='L-BFGS-B',
                        args=(temp,),
                        options={
                            'maxiter': 300,
                            'ftol': 1e-10,
                            'gtol': 1e-8
                        }
                    )
                    
                    opt_pts = result.x.reshape(N, 2)
                    opt_score = _score(opt_pts)
                    
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_pts.copy()
                except Exception:
                    pass
    
    if best_points is None:
        best_points = _initial_points_regular()
    
    return best_points
# EVOLVE-BLOCK-END
