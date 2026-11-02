import numpy as np
from scipy.optimize import minimize

N = 14

def points_from_params(params):
    """
    Convert params to points.
    params: [r_0, ..., r_13, alpha_0, ..., alpha_13]
    r_i: radii
    alpha_i: angular gaps between consecutive points
    """
    radii = params[:N]
    alphas = params[N:]
    
    # Ensure alphas are positive and sum to 2*pi
    # We will normalize alphas to sum to 2*pi
    alpha_sum = np.sum(alphas)
    if alpha_sum < 1e-9:
        alphas = np.full(N, 2 * np.pi / N)
    else:
        alphas = alphas / alpha_sum * (2 * np.pi)
        
    # Ensure radii are positive
    radii = np.abs(radii) + 1e-6
    
    # Compute angles
    # Fix theta_0 = 0
    thetas = np.zeros(N)
    for i in range(1, N):
        thetas[i] = thetas[i-1] + alphas[i-1]
    # Note: the last gap alphas[N-1] closes the loop to 2*pi
    
    x = radii * np.cos(thetas)
    y = radii * np.sin(thetas)
    return np.column_stack((x, y))

def convex_hull_area(points):
    """Calculate area using shoelace formula (points in CCW order)."""
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 distinct points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = abs(cross) * 0.5
                if area < min_area:
                    min_area = area
    return min_area

def is_convex_ccw(points):
    """Check if points form a strictly convex polygon in CCW order."""
    n = len(points)
    sign = None
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]
        cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
        if abs(cross) < 1e-9:
            return False
        if sign is None:
            sign = 1 if cross > 0 else -1
        else:
            if (cross > 0 and sign < 0) or (cross < 0 and sign > 0):
                return False
    return True

def objective(params):
    """Objective: minimize negative score."""
    points = points_from_params(params)
    
    if not is_convex_ccw(points):
        return 1e6
    
    area_hull = convex_hull_area(points)
    if area_hull < 1e-9:
        return 1e6
    
    area_min_tri = min_triangle_area(points)
    score = area_min_tri / area_hull
    return -score

def construct():
    N = 14
    best_score = -1e6
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Initial guess: Regular polygon
    base_radii = np.ones(N)
    base_alphas = np.full(N, 2 * np.pi / N)
    base_params = np.concatenate([base_radii, base_alphas])
    
    # Multi-start optimization
    for start_idx in range(10):
        if start_idx == 0:
            params0 = base_params.copy()
        else:
            # Perturb radii and alphas
            radii_pert = rng.normal(0, 0.1, N)
            alphas_pert = rng.normal(0, 0.1, N)
            params0 = np.concatenate([base_radii + radii_pert, base_alphas + alphas_pert])
            # Ensure alphas are positive for the initial guess
            params0[N:] = np.abs(params0[N:]) + 0.1
            params0[:N] = np.abs(params0[:N]) + 0.1
            
        result = minimize(
            objective,
            params0,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        
        final_params = result.x
        final_points = points_from_params(final_params)
        
        if is_convex_ccw(final_points):
            area_hull = convex_hull_area(final_points)
            if area_hull > 1e-9:
                area_min_tri = min_triangle_area(final_points)
                score = area_min_tri / area_hull
                if score > best_score:
                    best_score = score
                    best_points = final_points
    
    if best_points is None:
        # Fallback to regular polygon
        best_points = points_from_params(base_params)
    
    return best_points
