"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools

def _min_triangle_area_fast(points):
    """Compute the minimum area over all triples of points using vectorization."""
    n = len(points)
    # Compute all cross products for triples
    # For each pair (i,j), compute cross with all k
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            v1 = points[j] - points[i]
            # Cross product of v1 with all other vectors from i
            # v2_k = points[k] - points[i]
            # cross = v1[0]*v2_k[1] - v1[1]*v2_k[0]
            diffs = points - points[i]
            cross_vals = v1[0] * diffs[1] - v1[1] * diffs[0]
            # Only consider k > j
            areas = 0.5 * np.abs(cross_vals[j+1:])
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    pts = np.array(points)
    # Remove duplicates
    pts_unique = np.unique(pts, axis=0)
    n = len(pts_unique)
    if n <= 2:
        return 0.0
    
    # Andrew's monotone chain
    pts_sorted = pts_unique[np.lexsort((pts_unique[:, 1], pts_unique[:, 0]))]
    
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    
    lower = []
    for p in pts_sorted:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts_sorted):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    
    area = 0.0
    for i in range(len(hull)):
        j = (i + 1) % len(hull)
        area += hull[i][0]*hull[j][1] - hull[j][0]*hull[i][1]
    return abs(area) / 2.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    min_ta = _min_triangle_area_fast(points)
    hull_a = _convex_hull_area(points)
    if hull_a <= 0:
        return 0.0
    return min_ta / hull_a

def _build_points_10d(params):
    """
    params: 10-dim array.
    First 7: radial offsets for each heptagon vertex.
    Last 3: affine parameters [rotation, scale, shear].
    Returns 14 points.
    """
    offsets = params[:7]
    rotation = params[7]
    scale = params[8]
    shear = params[9]
    
    # Base heptagon points with radial offsets
    base_points = []
    for i in range(7):
        r1 = 1.0 + offsets[i]
        r2 = 1.0 - offsets[i]
        r1 = max(r1, 0.05)
        r2 = max(r2, 0.05)
        angle = i * 2 * np.pi / 7
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        base_points.append([r1 * cos_a, r1 * sin_a])
        base_points.append([r2 * cos_a, r2 * sin_a])
    
    base_points = np.array(base_points)
    
    # Apply affine transformation: scale, shear, rotation
    # Affine matrix: [[scale*cos(theta), -scale*sin(theta) + shear],
    #                [scale*sin(theta),  scale*cos(theta)]]
    # Simplified: rotation + uniform scale + shear
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    
    # Rotation matrix
    R = np.array([[cos_r, -sin_r],
                  [sin_r, cos_r]])
    
    # Shear matrix (horizontal shear)
    S = np.array([[1.0, shear],
                  [0.0, 1.0]])
    
    # Scale
    Sc = scale * np.eye(2)
    
    # Combined: R @ S @ Sc
    M = R @ S @ Sc
    
    points = base_points @ M.T
    
    return points

def _negative_score_10d(params):
    """Objective: minimize negative score for 10-parameter version."""
    points = _build_points_10d(params)
    score = _score(points)
    return -score

def construct():
    """
    Construct 14 points by optimizing 7 radial offsets + 3 affine parameters
    (rotation, scale, shear) for a regular heptagon where each vertex is split
    into two radially displaced points, then apply global affine transform.
    """
    np.random.seed(42)
    
    # Bounds: 7 offsets in [-0.5, 0.5], rotation in [0, 2*pi], scale in [0.5, 2.0], shear in [-1.0, 1.0]
    bounds = [(-0.5, 0.5)] * 7 + [(0, 2*np.pi), (0.5, 2.0), (-1.0, 1.0)]
    
    # Phase 1: Differential evolution for global optimization
    result = differential_evolution(
        _negative_score_10d, 
        bounds,
        seed=42,
        maxiter=150,
        tol=1e-6,
        popsize=12,
        mutation=(0.4, 1.0),
        recombination=0.7,
        polish=False
    )
    
    best_params = result.x.copy()
    best_score = -result.fun
    
    # Phase 2: Local refinement with Nelder-Mead
    try:
        res = minimize(_negative_score_10d, best_params, method='Nelder-Mead',
                      options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
        if -res.fun > best_score:
            best_params = res.x.copy()
            best_score = -res.fun
    except Exception:
        pass
    
    # Phase 3: Multi-start local optimization
    for trial in range(30):
        random_params = np.random.uniform(
            [-0.4]*7 + [0, 0.6, -0.8],
            [0.4]*7 + [2*np.pi, 1.8, 0.8]
        )
        try:
            res = minimize(_negative_score_10d, random_params, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11})
            if -res.fun > best_score:
                best_params = res.x.copy()
                best_score = -res.fun
        except Exception:
            pass
    
    points = _build_points_10d(best_params)
    
    # Verify validity
    hull_a = _convex_hull_area(points)
    if hull_a <= 0:
        # Fallback: regular heptagon with slight radial perturbation
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
