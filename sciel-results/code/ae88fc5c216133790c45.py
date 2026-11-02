import numpy as np
from scipy.optimize import minimize

N = 14

def points_from_radii(angles, radii):
    """Convert angles and radii to point coordinates."""
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
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

def objective(radii, angles):
    """Objective: minimize negative score."""
    points = points_from_radii(angles, radii)
    
    if not is_convex_ccw(points):
        return 1e6
    
    area_hull = convex_hull_area(points)
    if area_hull < 1e-9:
        return 1e6
    
    area_min_tri = min_triangle_area(points)
    score = area_min_tri / area_hull
    return -score

def construct():
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    best_score = -1e6
    best_points = None
    
    rng = np.random.default_rng(123)
    
    for start_idx in range(8):
        if start_idx == 0:
            angles = base_angles.copy()
        else:
            angle_perturb = rng.normal(0, 0.02 * (start_idx + 1), N)
            angles = base_angles + angle_perturb
            angles = np.sort(angles)
        
        if start_idx == 0:
            radii0 = np.ones(N)
        else:
            radii0 = 1.0 + rng.normal(0, 0.05 * (start_idx + 1), N)
        
        result = minimize(
            objective,
            radii0,
            args=(angles,),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        final_radii = result.x
        final_points = points_from_radii(angles, final_radii)
        
        if is_convex_ccw(final_points):
            area_hull = convex_hull_area(final_points)
            if area_hull > 1e-9:
                area_min_tri = min_triangle_area(final_points)
                score = area_min_tri / area_hull
                if score > best_score:
                    best_score = score
                    best_points = final_points
    
    if best_points is None:
        angles = base_angles
        radii = np.ones(N)
        best_points = points_from_radii(angles, radii)
    
    return best_points
