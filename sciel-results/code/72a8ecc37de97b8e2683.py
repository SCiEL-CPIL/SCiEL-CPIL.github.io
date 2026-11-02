"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 14

# EVOLVE-BLOCK-START
import random
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools


def _triangle_areas(points):
    """Compute all triangle areas for n points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                areas.append(area)
    return np.array(areas)


def _convex_hull_area(points):
    """Compute convex hull area."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except:
        # Fallback: use shoelace formula on sorted points
        pts = points.copy()
        # Find lowest y, then lowest x
        start = 0
        for i in range(1, len(pts)):
            if pts[i][1] < pts[start][1] or (pts[i][1] == pts[start][1] and pts[i][0] < pts[start][0]):
                start = i
        # Sort by polar angle from start
        def angle_cmp(p):
            return np.arctan2(p[1]-pts[start][1], p[0]-pts[start][0])
        sorted_pts = sorted(pts, key=angle_cmp)
        # Shoelace
        n = len(sorted_pts)
        area = 0.0
        for i in range(n):
            j = (i+1) % n
            area += sorted_pts[i][0]*sorted_pts[j][1]
            area -= sorted_pts[j][0]*sorted_pts[i][1]
        return abs(area)/2.0


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area


def _soft_min_objective(x, beta=50.0):
    """
    Objective to minimize: -log-sum-exp of triangle areas (approximates -min)
    plus penalty for convex hull area.
    """
    points = x.reshape(-1, 2)
    n = len(points)
    
    # Compute all triangle areas
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                areas.append(area)
    areas = np.array(areas)
    
    # Soft min via log-sum-exp
    max_a = np.max(areas)
    if max_a <= 1e-12:
        return 1e10
    
    # -log(sum(exp(beta * a))) / beta approximates -min(a) as beta -> inf
    # Use numerically stable version
    shifted = beta * (areas - max_a)
    log_sum = max_a * beta + np.log(np.sum(np.exp(shifted)))
    soft_min = -log_sum / beta
    
    # Convex hull area - we want to keep it reasonable
    hull_area = _convex_hull_area(points)
    
    # Objective: minimize -soft_min + penalty if hull area is too small or large
    # Normalize: we want min_area / hull_area to be large
    # Equivalent to maximizing soft_min / hull_area
    # So minimize -soft_min / hull_area
    
    if hull_area < 1e-6:
        return 1e10
    
    return -soft_min / hull_area


def _gradient(x, beta=50.0):
    """Numerical gradient for the objective."""
    eps = 1e-7
    grad = np.zeros_like(x)
    f0 = _soft_min_objective(x, beta)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        f_plus = _soft_min_objective(x_plus, beta)
        grad[i] = (f_plus - f0) / eps
    return grad


def _polish(points, max_iter=1000, tol=1e-10):
    """High-precision polish using scipy minimize."""
    x0 = points.flatten()
    
    # Try L-BFGS-B with numerical gradient
    try:
        result = minimize(
            _soft_min_objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 50}
        )
        if result.success or result.fun < _soft_min_objective(x0):
            new_points = result.x.reshape(-1, 2)
            return new_points
    except:
        pass
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_points = None
    best_score = 0.0
    
    rng = np.random.RandomState(42)
    
    # Strategy: multiple random restarts with optimization
    # Start with points in unit square, then optimize
    
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initial points in unit square
        points = rng.rand(N, 2)
        
        # Center and scale to unit area
        center = points.mean(axis=0)
        points = points - center
        # Scale so convex hull area is 1
        hull_area = _convex_hull_area(points)
        if hull_area > 1e-6:
            points = points / np.sqrt(hull_area)
        
        current_score = _score(points)
        
        # Optimize
        try:
            x0 = points.flatten()
            result = minimize(
                _soft_min_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.fun < _soft_min_objective(x0):
                new_points = result.x.reshape(-1, 2)
                new_score = _score(new_points)
                
                if new_score > current_score:
                    points = new_points
                    current_score = new_score
        except:
            pass
        
        # Polish
        if time.time() < deadline - 10:
            polished = _polish(points, max_iter=200, tol=1e-10)
            polished_score = _score(polished)
            if polished_score > current_score:
                points = polished
                current_score = polished_score
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        
        # Check time
        if time.time() > deadline:
            break
    
    # Final: ensure we return something valid
    if best_points is None:
        # Fallback: regular polygon
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
