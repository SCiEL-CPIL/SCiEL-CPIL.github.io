import numpy as np
from scipy.optimize import minimize
from itertools import combinations

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
    # Vectorized calculation for speed
    # We can compute all pairwise cross products? 
    # Number of triples is C(14,3) = 364. It's small enough to loop, but vectorizing is better.
    
    # Create indices
    # Loop is probably fine for 364 items, but let's try to be efficient.
    # The current loop is O(N^3). 14^3 = 2744. Very fast.
    # The bottleneck is likely the optimization iterations.
    
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

def generate_clustered_angles(epsilon=0.15):
    """
    Generate 14 angles clustered near the diagonals (pi/4, 3pi/4, etc).
    Base angles are uniform. We add a sinusoidal perturbation that
    increases density near the diagonals.
    """
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # sin(4*theta) has period pi/2.
    # At theta=pi/4, sin(pi)=0.
    # At theta=pi/8, sin(pi/2)=1.
    # If we add epsilon * sin(4*theta), points near pi/8 move towards pi/4?
    # Let's check derivative.
    # We want points to be CLOSER to pi/4.
    # If theta < pi/4, we want to increase it.
    # If theta > pi/4, we want to decrease it.
    # This means we want a term that is positive for theta in (0, pi/4) and negative for (pi/4, pi/2).
    # sin(4*theta) is positive in (0, pi/4) and negative in (pi/4, pi/2).
    # So adding epsilon * sin(4*theta) will push points towards pi/4.
    # Wait.
    # If theta = pi/8, sin(4*pi/8) = sin(pi/2) = 1. New theta = pi/8 + eps.
    # pi/8 = 22.5 deg. pi/4 = 45 deg.
    # So it moves AWAY from 0, towards 45? Yes.
    # If theta = 3*pi/8 (67.5 deg), sin(12*pi/8) = sin(1.5*pi) = -1.
    # New theta = 3*pi/8 - eps. Moves towards 45 deg.
    # So yes, this clusters points near the diagonals.
    
    angles = base_angles + epsilon * np.sin(4 * base_angles)
    # Ensure angles are sorted and within [0, 2pi)
    angles = np.sort(angles)
    # Handle wrap-around if any angle exceeds 2pi or goes below 0
    # With small epsilon, this shouldn't be an issue, but let's be safe.
    angles = angles % (2 * np.pi)
    angles = np.sort(angles)
    return angles

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
    # Generate clustered angles
    # Try a few different epsilon values to see if it helps
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2]
    
    best_score = -1e6
    best_points = None
    
    rng = np.random.default_rng(42)
    
    for eps in epsilons:
        if eps == 0.0:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        else:
            angles = generate_clustered_angles(eps)
        
        # Multi-start for radii
        for start_idx in range(5):
            if start_idx == 0:
                radii0 = np.ones(N)
            else:
                radii0 = 1.0 + rng.normal(0, 0.05, N)
            
            # Ensure initial radii are positive
            radii0 = np.abs(radii0) + 0.5
            
            result = minimize(
                objective,
                radii0,
                args=(angles,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            final_radii = result.x
            # Ensure final radii are positive
            final_radii = np.abs(final_radii) + 0.5
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
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.ones(N)
        best_points = points_from_radii(angles, radii)
    
    return best_points
