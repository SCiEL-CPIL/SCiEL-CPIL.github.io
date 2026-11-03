"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

# Deadline guard
_DEADLINE = time.time() + 130.0  # Safety margin before 150s kill

def _check_time():
    return time.time() < _DEADLINE

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas from n points."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                areas.append(area)
    return np.array(areas)

def _convex_hull_area(pts):
    """Compute area of convex hull of points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 1e-10

def _score(pts):
    """Minimum triangle area divided by convex hull area."""
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _soft_min_objective(x, beta):
    """
    Objective: minimize -soft_min(triangle_areas) / convex_hull_area
    We maximize min_area / hull_area, so we minimize -min_area / hull_area.
    Use log-sum-exp for smooth approximation of min.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Compute all triangle areas
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                areas.append(area)
    areas = np.array(areas)
    
    # Soft min using log-sum-exp trick: min(a) ≈ -(1/beta) * log(sum(exp(-beta*a)))
    # To avoid overflow, shift by max
    neg_areas = -areas
    max_neg = np.max(neg_areas)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    soft_min = -log_sum_exp / beta
    
    # Convex hull area
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    
    # We want to maximize soft_min / hull_area, so minimize negative
    return -soft_min / hull_area

def _objective_with_grad(x, beta):
    """Objective for L-BFGS-B with numerical gradient fallback."""
    return _soft_min_objective(x, beta)

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """Generate n points on a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])

def _concentric_rings(n):
    """Generate points on concentric rings for diverse initialization."""
    pts = []
    # One point at center
    pts.append([0.0, 0.0])
    # Distribute remaining points on rings
    remaining = n - 1
    if remaining <= 0:
        return np.array(pts)
    
    # Try different ring configurations
    ring_configs = [
        [remaining],  # All on one ring
        [remaining // 2, remaining - remaining // 2] if remaining >= 2 else [remaining],
    ]
    
    best_score = -1
    best_pts = None
    
    for config in ring_configs:
        pts_candidate = [[0.0, 0.0]]
        radii = [1.0] * len(config)
        for i, count in enumerate(config):
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            for ang in angles:
                pts_candidate.append([radii[i] * np.cos(ang), radii[i] * np.sin(ang)])
        
        if len(pts_candidate) != n:
            continue
        
        pts_arr = np.array(pts_candidate)
        s = _score(pts_arr)
        if s > best_score:
            best_score = s
            best_pts = pts_arr
    
    if best_pts is not None:
        return best_pts
    return _regular_polygon(n)

def _optimize_from_start(x0, beta_schedule):
    """Run L-BFGS-B from a starting point with progressive beta annealing."""
    current_x = x0.copy()
    
    for beta in beta_schedule:
        if not _check_time():
            break
        
        # Use L-BFGS-B
        try:
            result = minimize(
                _soft_min_objective,
                current_x,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < 1e10:
                current_x = result.x
        except Exception:
            pass
    
    return current_x

def _refine_with_local_search(pts):
    """Do a local refinement using Nelder-Mead on the raw score."""
    if not _check_time():
        return pts
    
    try:
        def neg_score(x):
            p = x.reshape(-1, 2)
            return -_score(p)
        
        result = minimize(
            neg_score,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        if result.fun < 0:  # Valid improvement
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return pts

def construct():
    """
    Place 13 points in a planar convex region to maximize the minimum area
    over all triples, normalized by the convex hull area.
    """
    global _DEADLINE
    _DEADLINE = time.time() + 130.0  # Reset deadline for this call
    
    best_pts = None
    best_score = -1
    
    # Generate diverse starting points
    starts = []
    
    # Regular polygon
    starts.append(_regular_polygon(N, radius=1.0))
    
    # Concentric rings
    starts.append(_concentric_rings(N))
    
    # Slightly perturbed regular polygons
    rng = np.random.RandomState(N)
    for _ in range(5):
        base = _regular_polygon(N, radius=1.0)
        perturbed = base + rng.randn(N, 2) * 0.1
        starts.append(perturbed)
    
    # Random points in unit disk
    for _ in range(3):
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = rng.uniform(0.3, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        starts.append(pts)
    
    # Beta schedule for progressive annealing
    beta_schedule = [20, 50, 100, 200]
    
    for x0 in starts:
        if not _check_time():
            break
        
        # Optimize with L-BFGS-B and beta annealing
        optimized = _optimize_from_start(x0.flatten(), beta_schedule)
        
        # Local refinement
        refined = _refine_with_local_search(optimized)
        
        s = _score(refined)
        if s > best_score:
            best_score = s
            best_pts = refined.copy()
    
    # If we haven't found anything good, use the best starting point
    if best_pts is None:
        for x0 in starts:
            s = _score(x0)
            if s > best_score:
                best_score = s
                best_pts = x0.copy()
    
    # Final check
    if best_pts is None:
        return _regular_polygon(N, radius=1.0)
    
    return best_pts
# EVOLVE-BLOCK-END
