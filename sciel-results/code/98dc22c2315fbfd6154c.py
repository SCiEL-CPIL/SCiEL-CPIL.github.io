"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar
from scipy.spatial import ConvexHull
import time
import math

N = 13

# EVOLVE-BLOCK-START

def _triangle_area(p, q, r):
    """Area of triangle formed by points p, q, r (2D vectors)."""
    return 0.5 * abs((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]))

def _min_triangle_area(points):
    """Minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area

def _convex_hull_area(points):
    """Area of the convex hull of points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, 'volume' is area
    except Exception:
        # Fallback: if all points are collinear or fewer than 3 distinct points
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _soft_min_triangle_area(points, beta=50.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp (soft-min).
    Returns a smooth approximation of the minimum triangle area.
    """
    n = len(points)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                areas[i, j, k] = a
                areas[j, i, k] = a
                areas[i, k, j] = a
                # We only need to consider unique triples, but for symmetry in gradient,
                # let's just collect all unique positive areas.
    
    # Collect unique triangle areas
    tri_areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tri_areas.append(_triangle_area(points[i], points[j], points[k]))
    
    tri_areas = np.array(tri_areas)
    if len(tri_areas) == 0:
        return 0.0
    
    # Soft-min: -log(sum(exp(-beta * x))) / beta
    # To avoid overflow, subtract max
    m = np.min(tri_areas)
    exp_terms = np.exp(-beta * (tri_areas - m))
    log_sum = np.log(np.sum(exp_terms))
    soft_min = m - log_sum / beta
    return soft_min

def _objective(params, beta=50.0):
    """
    Objective function for optimization.
    params: flattened array of Fourier coefficients.
    We parameterize points on a deformed regular 13-gon.
    To ensure convexity and avoid self-intersection, we use a radial deformation:
    r_k = 1 + sum_{m=1}^{M} (a_m cos(m theta_k) + b_m sin(m theta_k))
    where theta_k = 2 pi k / N.
    Points are (r_k cos(theta_k), r_k sin(theta_k)).
    We need to ensure r_k > 0. We can clamp or use exp.
    Let's use r_k = exp(sum...) to ensure positivity, or just clamp.
    Actually, for convexity, the radial function must be such that the polygon is convex.
    A regular polygon is convex. Small deformations preserve convexity.
    Let's use M=2 (up to 2nd harmonic) to keep it simple and ensure convexity.
    Number of params: 2 * M = 4.
    Plus a scaling factor? The score is scale-invariant (ratio), so scaling doesn't matter.
    However, the convex hull area changes with deformation.
    We optimize the shape.
    """
    M = 2
    a1, b1, a2, b2 = params
    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = 1.0 + a1 * np.cos(thetas) + b1 * np.sin(thetas) + a2 * np.cos(2 * thetas) + b2 * np.sin(2 * thetas)
    
    # Ensure r > 0 for convexity and validity
    # If r becomes <= 0, the point crosses the origin, likely non-convex.
    # We can penalize this heavily or clamp.
    # Let's clamp to a small positive value to maintain validity, but this creates non-differentiability.
    # Better: use a large penalty if min(r) < 0.1
    if np.min(r) < 0.1:
        return 1e6 - np.min(r)
    
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    points = np.column_stack((x, y))
    
    # We want to maximize min_triangle_area / hull_area
    # So we minimize -score
    # Using soft-min for smoothness
    min_tri = _soft_min_triangle_area(points, beta)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 1e6
    score = min_tri / hull_area
    return -score

def _objective_hard(params, beta=50.0):
    """
    Hard objective for final polishing or DE.
    Uses actual min triangle area.
    """
    M = 2
    a1, b1, a2, b2 = params
    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = 1.0 + a1 * np.cos(thetas) + b1 * np.sin(thetas) + a2 * np.cos(2 * thetas) + b2 * np.sin(2 * thetas)
    
    if np.min(r) < 0.1:
        return 1e6 - np.min(r)
    
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    points = np.column_stack((x, y))
    
    min_tri = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 1e6
    score = min_tri / hull_area
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial best: Regular 13-gon
    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x_init = np.cos(thetas)
    y_init = np.sin(thetas)
    points_init = np.column_stack((x_init, y_init))
    best_score = _score(points_init)
    best_points = points_init.copy()
    
    # Parameterization: a1, b1, a2, b2
    # Bounds: small deformations to ensure convexity
    # r = 1 + ...
    # If a1,b1 ~ 0.1, r is in [0.9, 1.1]. Convex.
    # If a2,b2 ~ 0.05, r changes less.
    bounds = [(-0.2, 0.2), (-0.2, 0.2), (-0.1, 0.1), (-0.1, 0.1)]
    
    # 1. Differential Evolution
    try:
        result_de = differential_evolution(
            _objective_hard,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            polish=True
        )
        if time.time() > deadline:
            return best_points
        
        params_de = result_de.x
        thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
        r = 1.0 + params_de[0] * np.cos(thetas) + params_de[1] * np.sin(thetas) + \
                params_de[2] * np.cos(2 * thetas) + params_de[3] * np.sin(2 * thetas)
        
        if np.min(r) > 0.1:
            x = r * np.cos(thetas)
            y = r * np.sin(thetas)
            points_de = np.column_stack((x, y))
            score_de = _score(points_de)
            if score_de > best_score:
                best_score = score_de
                best_points = points_de.copy()
    except Exception as e:
        pass
    
    if time.time() > deadline:
        return best_points
    
    # 2. Local Refinement using Nelder-Mead or L-BFGS-B on the hard objective
    # Start from best params found so far
    # We need to recover params from points? No, we kept params.
    # But if best_points came from init, params are 0,0,0,0.
    # Let's track best_params
    best_params = np.zeros(4)
    
    # Re-evaluate DE result params
    if 'params_de' in locals() and np.min(1.0 + params_de[0] * np.cos(np.linspace(0, 2*np.pi, N, endpoint=False)) + params_de[1] * np.sin(np.linspace(0, 2*np.pi, N, endpoint=False)) + params_de[2] * np.cos(2*np.linspace(0, 2*np.pi, N, endpoint=False)) + params_de[3] * np.sin(2*np.linspace(0, 2*np.pi, N, endpoint=False))) > 0.1:
         best_params = params_de
    
    # Try a few local optimizations from different starting points
    start_points = [
        np.zeros(4),
        best_params,
        np.array([0.05, 0, 0.02, 0]),
        np.array([0, 0.05, 0, 0.02]),
        np.array([-0.05, 0, 0.02, 0]),
    ]
    
    for sp in start_points:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                _objective_hard,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < -best_score:  # fun is -score
                score_new = -res.fun
                if score_new > best_score:
                    best_score = score_new
                    best_params = res.x
                    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
                    r = 1.0 + best_params[0] * np.cos(thetas) + best_params[1] * np.sin(thetas) + \
                            best_params[2] * np.cos(2 * thetas) + best_params[3] * np.sin(2 * thetas)
                    if np.min(r) > 0.1:
                        x = r * np.cos(thetas)
                        y = r * np.sin(thetas)
                        best_points = np.column_stack((x, y))
        except Exception:
            pass
    
    # 3. Final polish with higher beta soft-min and tighter tolerance?
    # Or just return best_points
    
    # Ensure the returned points are valid (convex hull area > 0)
    if _convex_hull_area(best_points) <= 1e-12:
        return points_init
        
    return best_points

# EVOLVE-BLOCK-END
