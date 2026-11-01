"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

DEADLINE = 110.0  # seconds, safety margin before 120s kill


def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of points using Andrew's monotone chain."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Sort by x, then by y
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Build lower hull
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    # Concatenate, removing duplicate endpoints
    hull = lower[:-1] + upper[:-1]
    
    if len(hull) < 3:
        return 0.0
    
    # Shoelace formula
    area = 0.0
    m = len(hull)
    for i in range(m):
        j = (i + 1) % m
        area += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1]
    return abs(area) * 0.5


def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _soft_min(values, temperature=0.01):
    """Soft minimum using log-sum-exp trick for smoothness."""
    values = np.array(values)
    min_val = np.min(values)
    # Use a large negative temperature for soft-min (approximates hard min)
    scaled = -values / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return min_val - temperature * (log_sum - max_scaled)


def _objective_smooth(x, temperature=0.01):
    """Smooth objective: negative soft-min triangle area / hull area."""
    points = x.reshape(-1, 2)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 1e10
    
    # Compute all triangle areas
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                areas.append(_triangle_area(points[i], points[j], points[k]))
    
    areas = np.array(areas)
    soft_min_area = _soft_min(areas, temperature)
    
    return -soft_min_area / hull_area


def _make_initial_heptagons(rotation=0.0, inner_scale=0.5, inner_rotation=0.0):
    """Create two concentric heptagons: outer at radius 1, inner at radius inner_scale."""
    outer_angles = rotation + np.linspace(0, 2*np.pi, 7, endpoint=False)
    inner_angles = inner_rotation + np.linspace(0, 2*np.pi, 7, endpoint=False)
    
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_pts = inner_scale * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    points = np.vstack([outer_pts, inner_pts])
    return points


def _perturb(points, magnitude=0.1):
    """Add random perturbation to points."""
    perturbed = points + magnitude * np.random.randn(*points.shape)
    return perturbed


def construct():
    start_time = time.time()
    
    # Precompute triangle indices for efficiency
    n = N
    tri_indices = list(combinations(range(n), 3))
    
    def _min_tri_fast(points):
        min_area = float('inf')
        for i, j, k in tri_indices:
            a = _triangle_area(points[i], points[j], points[k])
            if a < min_area:
                min_area = a
        return min_area
    
    def _score_fast(points):
        hull_area = _convex_hull_area(points)
        if hull_area <= 1e-12:
            return 0.0
        return _min_tri_fast(points) / hull_area
    
    best_score = 0.0
    best_points = None
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Config 1: Regular heptagon + smaller rotated heptagon
    for scale in [0.3, 0.4, 0.5, 0.6]:
        for rot_offset in [0, np.pi/7, 2*np.pi/7, np.pi/14]:
            for global_rot in [0, np.pi/14]:
                pts = _make_initial_heptagons(global_rot, scale, rot_offset + global_rot)
                initial_configs.append(pts)
    
    # Config 2: Perturbed versions
    np.random.seed(42)
    for _ in range(5):
        base = _make_initial_heptagons(0, 0.5, np.pi/7)
        initial_configs.append(_perturb(base, 0.05))
    
    # Config 3: Random points in unit circle
    for _ in range(3):
        angles = np.sort(np.random.uniform(0, 2*np.pi, n))
        radii = np.random.uniform(0.3, 1.0, n)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initial_configs.append(pts)
    
    # Evaluate all initial configs
    for idx, pts in enumerate(initial_configs):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        score = _score_fast(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Use Nelder-Mead to optimize the best initial configuration
    elapsed = time.time() - start_time
    if best_points is not None and elapsed < DEADLINE - 10:
        x0 = best_points.flatten()
        
        # Try multiple temperature values for the smooth objective
        for temp in [0.05, 0.02, 0.01]:
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 8:
                break
            
            try:
                result = minimize(
                    _objective_smooth,
                    x0,
                    args=(temp,),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'maxfev': 5000,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                
                if result.success or result.fun < -best_score:
                    optimized_pts = result.x.reshape(-1, 2)
                    opt_score = _score_fast(optimized_pts)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = optimized_pts.copy()
                        x0 = result.x  # Continue from improved point
            except Exception:
                pass
    
    # Final polish: local optimization with exact objective
    elapsed = time.time() - start_time
    if best_points is not None and elapsed < DEADLINE - 5:
        x0 = best_points.flatten()
        
        def _objective_exact(x):
            points = x.reshape(-1, 2)
            return -_score_fast(points)
        
        try:
            result = minimize(
                _objective_exact,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'maxfev': 8000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.fun < -best_score:
                optimized_pts = result.x.reshape(-1, 2)
                opt_score = _score_fast(optimized_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = optimized_pts.copy()
        except Exception:
            pass
    
    # Final check
    elapsed = time.time() - start_time
    if best_points is None:
        # Fallback: return a reasonable configuration
        best_points = _make_initial_heptagons(0, 0.5, np.pi/7)
    
    return best_points
# EVOLVE-BLOCK-END
