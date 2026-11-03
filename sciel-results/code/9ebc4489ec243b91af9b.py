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
    N = 14
    half_N = N // 2
    base_angles_full = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    best_score = -np.inf
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Multi-start with different symmetric angle perturbations
    for start_idx in range(10):
        # Generate symmetric angles
        # Perturb first half, mirror to second half to maintain reflectional symmetry
        perturb = rng.normal(0, 0.01 * (start_idx + 1), half_N)
        
        angles_full = np.zeros(N)
        for i in range(half_N):
            angles_full[i] = base_angles_full[i] + perturb[i]
            angles_full[N - 1 - i] = 2 * np.pi - angles_full[i]
        
        # Initial radii (symmetric)
        r_half_0 = np.ones(half_N)
        if start_idx > 0:
            r_half_0 += rng.normal(0, 0.05, half_N)
        
        # Objective function for half radii
        def obj_half(r_half):
            # Expand to full radii
            r_full = np.zeros(N)
            for i in range(half_N):
                r_full[i] = r_half[i]
                r_full[N - 1 - i] = r_half[i]
            
            points = points_from_radii(angles_full, r_full)
            
            if not is_convex_ccw(points):
                return 1e6
            
            area_hull = convex_hull_area(points)
            if area_hull < 1e-9:
                return 1e6
            
            area_min_tri = min_triangle_area(points)
            score = area_min_tri / area_hull
            return -score

        result = minimize(
            obj_half,
            r_half_0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        r_half_final = result.x
        r_full_final = np.zeros(N)
        for i in range(half_N):
            r_full_final[i] = r_half_final[i]
            r_full_final[N - 1 - i] = r_half_final[i]
        
        final_points = points_from_radii(angles_full, r_full_final)
        
        if is_convex_ccw(final_points):
            area_hull = convex_hull_area(final_points)
            if area_hull > 1e-9:
                area_min_tri = min_triangle_area(final_points)
                score = area_min_tri / area_hull
                if score > best_score:
                    best_score = score
                    best_points = final_points
    
    if best_points is None:
        angles = base_angles_full
        radii = np.ones(N)
        best_points = points_from_radii(angles, radii)
    
    return best_points
