import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 14

# Precompute triple indices for N=14
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _generate_points_superellipse(params):
    """
    Generate 14 points on a superellipse defined by |x/a|^n + |y/b|^n = 1.
    Points are distributed uniformly by arc length.
    params: [n, aspect_ratio]
    n: shape exponent (2 is circle, >2 is squarer, <2 is more pointed)
    aspect_ratio: a/b
    """
    n, aspect_ratio = params
    if n < 0.5 or n > 20.0:
        return None
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return None
        
    a = 1.0
    b = 1.0 / aspect_ratio
    
    # Normalize so that area is roughly 1? 
    # Area of superellipse = 4 * a * b * Gamma(1+1/n)^2 / Gamma(1+2/n)
    # We don't strictly need unit area for the score ratio, as the score is invariant to scaling.
    # But we need to ensure the points are well-distributed.
    
    # We will sample points by arc length.
    # Parametric form: x = a * sign(cos(t)) * |cos(t)|^(2/n), y = b * sign(sin(t)) * |sin(t)|^(2/n)
    # This parameterization does NOT have uniform arc length.
    # We need to reparameterize to uniform arc length.
    
    # 1. Generate many points to estimate arc length
    t_coarse = np.linspace(0, 2 * np.pi, 10000)
    x_c = a * np.sign(np.cos(t_coarse)) * np.abs(np.cos(t_coarse))**(2.0/n)
    y_c = b * np.sign(np.sin(t_coarse)) * np.abs(np.sin(t_coarse))**(2.0/n)
    
    # 2. Compute cumulative arc length
    dx = np.diff(x_c, prepend=x_c[-1])
    dy = np.diff(y_c, prepend=y_c[-1])
    ds = np.sqrt(dx**2 + dy**2)
    s_cum = np.cumsum(ds)
    s_total = s_cum[-1]
    s_cum = s_cum / s_total # Normalize to [0, 1]
    
    # 3. Interpolate to get 14 points with equal arc length spacing
    s_target = np.linspace(0, 1, N, endpoint=False)
    
    # Use linear interpolation for t given s
    # s_cum is monotonic increasing
    t_fine = np.interp(s_target, s_cum, t_coarse)
    
    # Calculate final points
    x = a * np.sign(np.cos(t_fine)) * np.abs(np.cos(t_fine))**(2.0/n)
    y = b * np.sign(np.sin(t_fine)) * np.abs(np.sin(t_fine))**(2.0/n)
    
    points = np.column_stack((x, y))
    return points

def _score_params(params):
    """
    Objective function for optimization.
    params: [n, aspect_ratio]
    """
    points = _generate_points_superellipse(params)
    if points is None:
        return 0.0
    return _calculate_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # 1. Global Search using Differential Evolution
    # n in [1.0, 5.0], aspect_ratio in [0.8, 1.25] (reciprocal covers the rest)
    # Actually, due to symmetry, aspect ratio > 1 is equivalent to < 1 with rotation.
    # But superellipse definition is symmetric. Let's search n in [1, 5] and ar in [0.8, 1.25].
    # Wait, if ar > 1, it's just rotated 90 degrees? No, x and y are swapped in definition essentially.
    # The set of points for (n, ar) is the same as (n, 1/ar) rotated by 90 degrees.
    # So we only need ar in [1.0, 2.0] or [0.5, 1.0]. Let's use [1.0, 1.5].
    
    bounds = [(1.0, 5.0), (1.0, 1.5)]
    
    best_score = 0.0
    best_params = None
    
    if time.time() < deadline - 30:
        try:
            res = differential_evolution(
                lambda p: -_score_params(p),
                bounds,
                seed=42,
                maxiter=50,
                tol=1e-4,
                polish=False
            )
            best_score = -res.fun
            best_params = res.x
        except Exception:
            pass
            
    if best_params is None:
        # Fallback to circle
        best_params = [2.0, 1.0]
        best_score = _score_params(best_params)

    # 2. Local Refinement using Nelder-Mead
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lambda p: -_score_params(p),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500}
            )
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
            
    # 3. Final Polish with Powell (more robust for non-smooth minima)
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda p: -_score_params(p),
                best_params,
                method='Powell',
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass

    # Generate final points
    points = _generate_points_superellipse(best_params)
    
    # Fallback if something went wrong
    if points is None or len(points) != N:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return points
