"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Area of triangle = 0.5 * |cross(pj-pi, pk-pi)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    # Use a simple convex hull algorithm (Andrew's monotone chain)
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

def _regular_heptagon_base():
    """Generate 7 points on a regular heptagon (unit circle)."""
    angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _build_points_from_params(params):
    """
    params: 10-dim array
    [0:7] = radial offsets for 7 heptagon vertices (each vertex gets 2 points: one at radius 1+offset, one at radius 1-offset)
    [7:10] = affine transformation parameters (rotation angle, scale, shear)
    """
    offsets = params[:7]
    rot = params[7]
    scale = params[8]
    shear = params[9]
    
    base = _regular_heptagon_base()
    
    points = []
    for i in range(7):
        r1 = 1.0 + offsets[i]
        r2 = 1.0 - offsets[i]
        # Ensure radii are positive
        r1 = max(r1, 0.1)
        r2 = max(r2, 0.1)
        angle = i * 2 * np.pi / 7
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        p1 = np.array([r1 * cos_a, r1 * sin_a])
        p2 = np.array([r2 * cos_a, r2 * sin_a])
        points.append(p1)
        points.append(p2)
    
    points = np.array(points)
    
    # Apply affine transformation: rotation, scale, shear
    # Affine matrix: [[scale*cos(rot), -scale*sin(rot) + shear],
    #                [scale*sin(rot),  scale*cos(rot)]]
    c = np.cos(rot)
    s = np.sin(rot)
    A = np.array([[scale * c, -scale * s + shear],
                  [scale * s,  scale * c]])
    
    points = points @ A.T
    
    return points

def _negative_score(params):
    """Objective function for minimization (we minimize negative score)."""
    points = _build_points_from_params(params)
    score = _score(points)
    return -score

def construct():
    """
    Construct 14 points by optimizing 10 parameters:
    7 radial offsets for a regular heptagon (each vertex split into two points)
    + 3 affine parameters (rotation, scale, shear).
    """
    # Initial guess: all offsets zero, no rotation, scale=1, no shear
    x0 = np.zeros(10)
    x0[7] = 0.0  # rotation
    x0[8] = 1.0  # scale
    x0[9] = 0.0  # shear
    
    # Try multiple random initializations and keep the best
    best_score = -float('inf')
    best_params = x0.copy()
    
    np.random.seed(42)
    
    # First try the zero initialization
    points = _build_points_from_params(x0)
    score = _score(points)
    if score > best_score:
        best_score = score
        best_params = x0.copy()
    
    # Try several random initializations
    for trial in range(50):
        offsets = np.random.uniform(-0.3, 0.3, 7)
        rot = np.random.uniform(0, 2*np.pi)
        scale = np.random.uniform(0.5, 2.0)
        shear = np.random.uniform(-0.5, 0.5)
        
        x_trial = np.concatenate([offsets, [rot, scale, shear]])
        
        # Quick evaluation
        points_trial = _build_points_from_params(x_trial)
        score_trial = _score(points_trial)
        
        if score_trial > best_score:
            best_score = score_trial
            best_params = x_trial.copy()
        
        # Run a local optimization from this starting point
        try:
            res = minimize(_negative_score, x_trial, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.success or res.fun < -best_score:
                score_res = -res.fun
                if score_res > best_score:
                    best_score = score_res
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish with a more careful optimization
    try:
        res = minimize(_negative_score, best_params, method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        if -res.fun > best_score:
            best_params = res.x.copy()
            best_score = -res.fun
    except Exception:
        pass
    
    points = _build_points_from_params(best_params)
    
    # Ensure the points form a valid convex region (convex hull area > 0)
    hull_a = _convex_hull_area(points)
    if hull_a <= 0:
        # Fallback: regular heptagon with duplicated points slightly perturbed
        base = _regular_heptagon_base()
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
