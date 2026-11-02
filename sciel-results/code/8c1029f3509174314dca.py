"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# EVOLVE-BLOCK-START

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples of points."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs(
                    (pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1])
                    - (pts[k][0] - pts[i][0]) * (pts[j][1] - pts[i][1])
                )
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(pts):
    """Compute area of convex hull of points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    min_ta = _min_triangle_area(pts)
    return min_ta / hull_area


def _soft_min_triangle_area(pts, beta=50.0):
    """Differentiable soft-min of triangle areas."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs(
                    (pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1])
                    - (pts[k][0] - pts[i][0]) * (pts[j][1] - pts[i][1])
                )
                areas.append(area)
    areas = np.array(areas)
    # Soft min: -log-sum-exp / beta approximates min
    m = np.min(areas)
    if m < 1e-15:
        return m
    return m - np.log(np.sum(np.exp(-beta * (areas - m)))) / beta


def _objective(params, beta=50.0):
    """Objective to minimize: -soft_min_triangle_area / hull_area + penalty for non-convexity."""
    pts = params.reshape(-1, 2)
    # Convexity penalty: check that all points are on convex hull
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 1e6  # Degenerate
    
    if hull_area < 1e-8:
        return 1e6
    
    # Penalty if any point is not on the hull (interior points)
    hull_indices = set(hull.vertices)
    n_interior = N - len(hull_indices)
    if n_interior > 0:
        return 1e6 + n_interior * 100.0
    
    soft_min = _soft_min_triangle_area(pts, beta)
    score = soft_min / hull_area
    return -score


def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """Generate a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])


def _ring_config(n, n_ring, radius_inner=0.3, radius_outer=1.0, phase=0.0):
    """Generate points on two concentric rings."""
    pts = []
    # Outer ring
    n_outer = n - n_ring
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + phase
    for a in angles_outer:
        pts.append([radius_outer * np.cos(a), radius_outer * np.sin(a)])
    # Inner ring
    angles_inner = np.linspace(0, 2 * np.pi, n_ring, endpoint=False) + phase + np.pi / n_ring
    for a in angles_inner:
        pts.append([radius_inner * np.cos(a), radius_inner * np.sin(a)])
    return np.array(pts)


def _elliptic_config(n, a=1.0, b=0.8, phase=0.0):
    """Generate points on an ellipse."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase
    return np.column_stack([a * np.cos(angles), b * np.sin(angles)])


def _random_convex(n, rng):
    """Generate random convex position points by sorting random angles."""
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    radii = rng.uniform(0.5, 1.0, n)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def construct():
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    best_pts = _regular_polygon(N)
    best_score = _score(best_pts)
    
    # Generate diverse initial configurations
    inits = []
    
    # Regular polygon
    inits.append(_regular_polygon(N))
    
    # Elliptic variations
    for a in [1.0, 1.2, 0.8]:
        for b in [0.7, 0.9, 1.1]:
            for phase in [0.0, np.pi / N, np.pi / (2 * N)]:
                inits.append(_elliptic_config(N, a, b, phase))
    
    # Ring configurations
    for n_ring in [3, 4, 5, 6]:
        for r_in in [0.2, 0.35, 0.5]:
            for phase in [0.0, np.pi / N]:
                inits.append(_ring_config(N, n_ring, r_in, 1.0, phase))
    
    # Random convex
    rng = np.random.RandomState(42)
    for _ in range(10):
        inits.append(_random_convex(N, rng))
    
    # Evaluate all initial configs
    for init in inits:
        if time.time() > deadline - 30:
            break
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init.copy()
    
    # Local optimization with L-BFGS-B from best initial configs
    # Sort initial configs by score
    scored_inits = []
    for init in inits:
        s = _score(init)
        scored_inits.append((s, init))
    scored_inits.sort(key=lambda x: -x[0])
    
    # Optimize top initial configurations
    max_optimizations = 15
    count = 0
    for s_init, init in scored_inits:
        if time.time() > deadline - 20:
            break
        if count >= max_optimizations:
            break
        
        x0 = init.flatten()
        
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 5000}
            )
            
            if result.success or result.fun < 0:
                opt_pts = result.x.reshape(-1, 2)
                s_opt = _score(opt_pts)
                if s_opt > best_score:
                    best_score = s_opt
                    best_pts = opt_pts.copy()
        except Exception:
            pass
        
        count += 1
    
    # Additional fine-tuning with higher beta from the best point
    if time.time() < deadline - 15:
        x0 = best_pts.flatten()
        for beta in [100.0, 200.0]:
            if time.time() > deadline - 10:
                break
            try:
                def obj_high_beta(params):
                    pts = params.reshape(-1, 2)
                    try:
                        hull = ConvexHull(pts)
                        hull_area = hull.volume
                    except Exception:
                        return 1e6
                    if hull_area < 1e-8:
                        return 1e6
                    hull_indices = set(hull.vertices)
                    if N - len(hull_indices) > 0:
                        return 1e6
                    soft_min = _soft_min_triangle_area(pts, beta)
                    return -soft_min / hull_area
                
                result = minimize(
                    obj_high_beta,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'maxfun': 3000}
                )
                
                if result.success or result.fun < 0:
                    opt_pts = result.x.reshape(-1, 2)
                    s_opt = _score(opt_pts)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_pts = opt_pts.copy()
            except Exception:
                pass
    
    return best_pts

# EVOLVE-BLOCK-END
