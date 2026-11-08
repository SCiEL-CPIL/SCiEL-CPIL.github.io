"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import sys

def _all_triangle_areas(points):
    """Compute all triangle areas for all triples."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                areas.append(area)
    return np.array(areas)

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    return float(np.min(_all_triangle_areas(points)))

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _soft_score(points, beta=50.0):
    """Smooth surrogate: -log-sum-exp of negative triangle areas, normalized by hull area.
    This approximates the min triangle area with a smooth function."""
    areas = _all_triangle_areas(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    # Use log-sum-exp trick for numerical stability
    a = -areas / hull_area
    a_max = np.max(a)
    lse = a_max + np.log(np.sum(np.exp(a - a_max)))
    return -lse / beta  # Negative so we minimize

def _neg_soft_score(x, beta=50.0):
    """Objective for minimization: negative soft score (we want to maximize soft score)."""
    points = x.reshape(N, 2)
    return -_soft_score(points, beta)

def _regular_polygon(n, radius=1.0):
    """Generate points on a regular polygon."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _two_rings(n, inner_frac=0.5):
    """Generate points on two concentric rings."""
    n_outer = n // 2
    n_inner = n - n_outer
    
    outer_angles = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
    inner_angles = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + np.pi/n_inner
    
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_pts = inner_frac * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    return np.vstack([outer_pts, inner_pts])

def _hexagonal_packing(n):
    """Generate points in a hexagonal packing pattern."""
    points = []
    # Center
    points.append([0, 0])
    # First ring (6 points)
    for i in range(6):
        angle = i * np.pi / 3
        points.append([np.cos(angle), np.sin(angle)])
    # Second ring (7 points to make 14 total)
    for i in range(7):
        angle = i * 2 * np.pi / 7 + np.pi/7
        points.append([2 * np.cos(angle), 2 * np.sin(angle)])
    return np.array(points)

def _optimize_powell(points, deadline=None, beta=50.0):
    """Optimize using Powell's method on the smooth surrogate objective."""
    x0 = points.flatten()
    
    # Define callback to check time
    time_check = {'stop': False}
    def callback(xk):
        if deadline is not None and time.time() > deadline:
            time_check['stop'] = True
    
    try:
        result = minimize(
            _neg_soft_score,
            x0,
            method='Powell',
            args=(beta,),
            options={'maxiter': 2000, 'maxfev': 50000, 'xatol': 1e-8, 'fatol': 1e-10},
            callback=callback
        )
        opt_points = result.x.reshape(N, 2)
    except Exception:
        opt_points = points.copy()
    
    return opt_points

def _polish_exact(points, deadline=None):
    """Fine-tune using exact score with small perturbations."""
    best_points = points.copy()
    best_score = _score(points)
    step = 0.01
    n = N
    
    for it in range(2000):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = np.random.randint(n)
        delta = np.random.normal(0, step, 2)
        new_points = best_points.copy()
        new_points[idx] += delta
        
        new_score = _score(new_points)
        if new_score > best_score:
            best_score = new_score
            best_points = new_points
        else:
            if it % 200 == 0:
                step *= 0.85
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin (5s buffer before 140s, 10s before 145s)
    
    # Initialize with several different configurations
    candidates = []
    
    # Regular polygon
    candidates.append(_regular_polygon(N))
    
    # Two rings with different ratios
    candidates.append(_two_rings(N, 0.4))
    candidates.append(_two_rings(N, 0.5))
    candidates.append(_two_rings(N, 0.6))
    candidates.append(_two_rings(N, 0.7))
    
    # Hexagonal packing
    candidates.append(_hexagonal_packing(N))
    
    # Random initializations
    rng = np.random.RandomState(42)
    for _ in range(8):
        candidates.append(rng.rand(N, 2))
    
    # Evaluate initial candidates
    best_points = None
    best_score = -1
    
    for pts in candidates:
        if time.time() > deadline:
            break
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Powell optimization from multiple starting points
    for i, pts in enumerate(candidates):
        if time.time() > deadline - 5:
            break
        # Skip if too close to current best
        if best_points is not None and np.allclose(pts, best_points, atol=1e-6):
            continue
        
        opt_pts = _optimize_powell(pts, deadline=deadline)
        opt_score = _score(opt_pts)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts.copy()
    
    # Final exact polish
    if best_points is not None and time.time() < deadline - 3:
        best_points, best_score = _polish_exact(
            best_points, 
            deadline=deadline
        )
    
    return best_points if best_points is not None else _regular_polygon(N)
# EVOLVE-BLOCK-END
