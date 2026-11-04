"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 1:
        return 0.0
    if len(pts) == 2:
        return 0.0
    
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    
    area = 0.0
    for i in range(len(hull)):
        j = (i + 1) % len(hull)
        area += hull[i][0]*hull[j][1] - hull[j][0]*hull[i][1]
    return abs(area) / 2.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    min_ta = _min_triangle_area(points)
    hull_a = _convex_hull_area(points)
    if hull_a <= 0:
        return 0.0
    return min_ta / hull_a

def _build_points(params):
    """
    params: 10-dim array.
    First 7: radial offsets for each of 7 heptagon directions.
    Next 3: affine parameters (rotation, scale_x, scale_y) applied to base heptagon.
    
    Base: regular heptagon on unit circle.
    Apply affine transform: rotate by theta, scale x by sx, y by sy.
    Then for each direction i, place two points at radii (1+d_i) and (1-d_i) 
    along the transformed radial direction.
    
    Returns 14 points.
    """
    offsets = params[:7]
    theta = params[7]
    sx = params[8]
    sy = params[9]
    
    # Ensure positive scales
    sx = max(sx, 0.3)
    sy = max(sy, 0.3)
    
    points = []
    for i in range(7):
        base_angle = i * 2 * np.pi / 7
        # Rotate
        ang = base_angle + theta
        cos_a = np.cos(ang)
        sin_a = np.sin(ang)
        
        # Scale the radial direction
        dir_x = sx * cos_a
        dir_y = sy * sin_a
        
        # Two points at radii 1+offset and 1-offset along this direction
        r1 = 1.0 + offsets[i]
        r2 = 1.0 - offsets[i]
        r1 = max(r1, 0.05)
        r2 = max(r2, 0.05)
        
        points.append([r1 * dir_x, r1 * dir_y])
        points.append([r2 * dir_x, r2 * dir_y])
    
    return np.array(points)

def _negative_score(params):
    """Objective: minimize negative score."""
    points = _build_points(params)
    score = _score(points)
    return -score

def construct():
    """
    Construct 14 points by optimizing 7 radial offsets and 3 affine parameters
    for a regular heptagon where each vertex is split into two radially displaced points.
    """
    np.random.seed(123)
    
    # Define bounds: 7 offsets in [-0.4, 0.4], theta in [-pi/7, pi/7], sx,sy in [0.5, 2.0]
    bounds = [(-0.4, 0.4)] * 7 + [(-np.pi/7, np.pi/7), (0.5, 2.0), (0.5, 2.0)]
    
    # Use differential evolution for global optimization
    result = differential_evolution(
        _negative_score, 
        bounds,
        seed=42,
        maxiter=300,
        tol=1e-10,
        popsize=20,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True
    )
    
    best_params = result.x
    best_score = -result.fun
    
    # Additional local refinement with Nelder-Mead
    try:
        res = minimize(_negative_score, best_params, method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
        if -res.fun > best_score:
            best_params = res.x
            best_score = -res.fun
    except Exception:
        pass
    
    # Try multiple random starts with local optimization
    for trial in range(30):
        random_params = np.array([
            np.random.uniform(-0.3, 0.3) for _ in range(7)
        ] + [
            np.random.uniform(-np.pi/7, np.pi/7),
            np.random.uniform(0.7, 1.5),
            np.random.uniform(0.7, 1.5)
        ])
        try:
            res = minimize(_negative_score, random_params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11})
            if -res.fun > best_score:
                best_params = res.x
                best_score = -res.fun
        except Exception:
            pass
    
    points = _build_points(best_params)
    
    # Verify validity
    hull_a = _convex_hull_area(points)
    if hull_a <= 0:
        # Fallback: symmetric heptagon
        points = []
        for i in range(7):
            angle = i * 2 * np.pi / 7
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            points.append([1.1 * cos_a, 1.1 * sin_a])
            points.append([0.9 * cos_a, 0.9 * sin_a])
        points = np.array(points)
    
    return points
# EVOLVE-BLOCK-END
