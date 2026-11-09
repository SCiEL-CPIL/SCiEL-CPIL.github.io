"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
DEADLINE = 120.0  # seconds wall-clock limit; keep safety margin


def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = float('inf')
    # Vectorize over all triples using broadcasting
    # indices for all C(n,3) triples
    idx = np.array(list(combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)
    return min_area


def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0
    ma = _min_triangle_area(points)
    return ma / ha


def _negative_score(points):
    return -_score(points)


def _make_12gon_center():
    """Create a regular 12-gon with a center point, normalized to unit hull area."""
    # Regular 12-gon vertices on unit circle
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    # Add center
    pts = np.vstack([pts, np.array([0.0, 0.0])])
    # Normalize so hull area = 1
    ha = _hull_area(pts)
    if ha > 1e-12:
        scale = 1.0 / np.sqrt(ha)
        pts = pts * scale
    return pts


def _make_13gon():
    """Create a regular 13-gon normalized to unit hull area."""
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    ha = _hull_area(pts)
    if ha > 1e-12:
        scale = 1.0 / np.sqrt(ha)
        pts = pts * scale
    return pts


def _optimize_from_start(start_points, max_time=60.0):
    """Run Nelder-Mead optimization from a starting configuration."""
    t0 = time.time()
    x0 = start_points.flatten()
    
    options = {
        'maxiter': 5000,
        'xatol': 1e-10,
        'fatol': 1e-12,
        'adaptive': True,
    }
    
    result = minimize(
        _negative_score,
        x0,
        method='Nelder-Mead',
        options=options
    )
    
    # Check deadline
    if time.time() - t0 > max_time:
        return start_points  # return original if timeout during opt
    
    opt_points = result.x.reshape(-1, 2)
    return opt_points


def construct():
    """Construct 13 points to maximize min triangle area / hull area."""
    t_start = time.time()
    deadline = t_start + DEADLINE - 5.0  # 5s safety margin
    
    best_points = None
    best_score = 0.0
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. Regular 12-gon + center
    starts.append(_make_12gon_center())
    
    # 2. Regular 13-gon
    starts.append(_make_13gon())
    
    # 3. 12-gon + center with slight perturbation
    for _ in range(3):
        base = _make_12gon_center()
        noise = np.random.randn(13, 2) * 0.05
        perturbed = base + noise
        starts.append(perturbed)
    
    # 4. 13-gon with perturbation
    for _ in range(2):
        base = _make_13gon()
        noise = np.random.randn(13, 2) * 0.05
        perturbed = base + noise
        starts.append(perturbed)
    
    # 5. Random points in unit square
    for _ in range(2):
        rand_pts = np.random.rand(13, 2)
        starts.append(rand_pts)
    
    # 6. 11-gon + 2 interior points
    angles = np.linspace(0, 2 * np.pi, 11, endpoint=False)
    pts11 = np.column_stack([np.cos(angles), np.sin(angles)])
    interior = np.array([[0.3, 0.3], [-0.3, -0.3]])
    combined = np.vstack([pts11, interior])
    ha = _hull_area(combined)
    if ha > 1e-12:
        combined = combined / np.sqrt(ha)
    starts.append(combined)
    
    # Evaluate each start and optimize
    for i, start in enumerate(starts):
        if time.time() > deadline:
            break
        
        # Evaluate initial score
        s = _score(start)
        if s > best_score:
            best_score = s
            best_points = start.copy()
        
        # Check if we have enough time for optimization
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
        
        # Run optimization with time budget
        max_opt_time = min(remaining - 5.0, 40.0)
        opt_points = _optimize_from_start(start, max_time=max_opt_time)
        
        s_opt = _score(opt_points)
        if s_opt > best_score:
            best_score = s_opt
            best_points = opt_points.copy()
        
        # If good score, try a second round of optimization from the result
        if s_opt > best_score * 0.95 and time.time() < deadline - 15.0:
            remaining2 = deadline - time.time() - 5.0
            if remaining2 > 10.0:
                opt2 = _optimize_from_start(opt_points, max_time=min(remaining2, 30.0))
                s_opt2 = _score(opt2)
                if s_opt2 > best_score:
                    best_score = s_opt2
                    best_points = opt2.copy()
    
    # If no points found (shouldn't happen), return a fallback
    if best_points is None:
        best_points = _make_12gon_center()
    
    return best_points
