import numpy as np
from scipy.optimize import minimize

N = 14

def support_to_points(coefficients, angles):
    """
    Convert support function coefficients to convex polygon vertices.
    The support function h(θ) defines the distance from origin to supporting line.
    For a convex polygon, the vertices are intersections of adjacent support lines.
    
    Given angles θ_i and coefficients h_i = h(θ_i), the vertex between 
    supporting lines at θ_i and θ_{i+1} is found by intersecting:
      x·cos(θ_i) + y·sin(θ_i) = h_i
      x·cos(θ_{i+1}) + y·sin(θ_{i+1}) = h_{i+1}
    """
    n = len(angles)
    vertices = np.zeros((n, 2))
    
    for i in range(n):
        j = (i + 1) % n
        θ1, θ2 = angles[i], angles[j]
        h1, h2 = coefficients[i], coefficients[j]
        
        c1, s1 = np.cos(θ1), np.sin(θ1)
        c2, s2 = np.cos(θ2), np.sin(θ2)
        
        det = c1 * s2 - c2 * s1
        if abs(det) < 1e-12:
            # Lines are parallel, use midpoint as fallback
            vertices[i] = 0.5 * (h1 * np.array([c1, s1]) + h2 * np.array([c2, s2]))
        else:
            # Solve: c1*x + s1*y = h1
            #         c2*x + s2*y = h2
            x = (h1 * s2 - h2 * s1) / det
            y = (c1 * h2 - c2 * h1) / det
            vertices[i] = np.array([x, y])
    
    return vertices

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

def objective(coefficients, angles):
    """Objective: minimize negative score using support function representation."""
    # Ensure all coefficients are positive (convexity requirement)
    if np.any(coefficients <= 1e-6):
        return 1e6
    
    points = support_to_points(coefficients, angles)
    
    if not is_convex_ccw(points):
        return 1e6
    
    area_hull = convex_hull_area(points)
    if area_hull < 1e-9:
        return 1e6
    
    area_min_tri = min_triangle_area(points)
    score = area_min_tri / area_hull
    return -score

def generate_non_uniform_angles():
    """
    Generate non-uniform angles that cluster near the diagonals (π/4, 3π/4, 5π/4, 7π/4).
    This better resolves the geometry of the smallest triangles.
    """
    # Base uniform angles
    base = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Add clustering near diagonals
    diagonal_angles = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
    
    # Create additional sample points near diagonals
    extra_angles = []
    for d in diagonal_angles:
        for offset in [-0.15, 0.15]:
            extra_angles.append(d + offset)
    
    # Combine and sort
    all_angles = np.concatenate([base, extra_angles])
    all_angles = np.sort(all_angles)
    
    # Select 14 angles with higher density near diagonals
    # Use a weighted selection: keep all base angles but adjust some to be near diagonals
    angles = base.copy()
    
    # Perturb angles to cluster near diagonals
    for i in range(N):
        # Find nearest diagonal
        dists = np.abs(angles[i] - diagonal_angles)
        nearest_diag = diagonal_angles[np.argmin(dists)]
        
        # Blend toward diagonal with some randomness
        blend = 0.3
        angles[i] = (1 - blend) * angles[i] + blend * nearest_diag
    
    angles = np.sort(angles)
    return angles

def construct():
    angles = generate_non_uniform_angles()
    
    best_score = -1e6
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Multi-start optimization
    n_starts = 12
    
    for start_idx in range(n_starts):
        if start_idx == 0:
            # Start with circle (all coefficients equal)
            coefficients0 = np.ones(N)
        elif start_idx == 1:
            # Start with slight radial perturbation
            coefficients0 = 1.0 + rng.normal(0, 0.05, N)
        elif start_idx == 2:
            # Start with alternating pattern
            coefficients0 = 1.0 + 0.1 * np.sin(2 * angles)
        elif start_idx == 3:
            # Start with cosine pattern
            coefficients0 = 1.0 + 0.1 * np.cos(3 * angles)
        else:
            # Random starts with varying magnitudes
            scale = 0.05 + 0.05 * start_idx
            coefficients0 = 1.0 + rng.normal(0, scale, N)
        
        # Ensure positive
        coefficients0 = np.maximum(coefficients0, 0.5)
        
        result = minimize(
            objective,
            coefficients0,
            args=(angles,),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
        )
        
        final_coefficients = result.x
        
        # Check if valid
        if np.any(final_coefficients <= 1e-6):
            continue
        
        final_points = support_to_points(final_coefficients, angles)
        
        if is_convex_ccw(final_points):
            area_hull = convex_hull_area(final_points)
            if area_hull > 1e-9:
                area_min_tri = min_triangle_area(final_points)
                score = area_min_tri / area_hull
                if score > best_score:
                    best_score = score
                    best_points = final_points
    
    if best_points is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        coefficients = np.ones(N)
        best_points = support_to_points(coefficients, angles)
    
    return best_points
