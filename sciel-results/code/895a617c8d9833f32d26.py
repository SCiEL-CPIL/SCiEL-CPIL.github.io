import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# Pre-compute indices for all triangles for n=13
# This is static for the task
_ALL_TRI_INDICES = np.array(list(combinations(range(N), 3)), dtype=int)

def _score_configuration(points):
    """
    Calculate the score for a given set of points.
    Score = min_triangle_area / convex_hull_area
    Vectorized for speed.
    """
    points = np.array(points, dtype=float)
    n = len(points)
    if n < 3:
        return 0.0
    
    # Check for degenerate hull
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area <= 1e-10:
        return 0.0
        
    # Vectorized triangle area calculation
    # Get all combinations of 3 indices
    # _ALL_TRI_INDICES has shape (286, 3)
    # points[_ALL_TRI_INDICES] would be shape (286, 3, 2)
    
    p1 = points[_ALL_TRI_INDICES[:, 0]]
    p2 = points[_ALL_TRI_INDICES[:, 1]]
    p3 = points[_ALL_TRI_INDICES[:, 2]]
    
    # Cross product magnitude
    # area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    d1 = p2 - p1
    d2 = p3 - p1
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area < 1e-12:
        return 0.0

    return min_area / hull_area

def _generate_concentric_points(n1, r1, phi1, n2, r2, phi2, include_center=True):
    """
    Generate points for a configuration with up to 2 concentric rings and an optional center.
    Total points must be 13.
    If include_center: 1 + n1 + n2 = 13 => n1 + n2 = 12
    If not include_center: n1 + n2 = 13
    """
    points = []
    if include_center:
        points.append([0.0, 0.0])
    
    if n1 > 0:
        angles1 = np.arange(n1) * (2 * np.pi / n1) + phi1
        pts1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
        points.extend(pts1)
        
    if n2 > 0:
        angles2 = np.arange(n2) * (2 * np.pi / n2) + phi2
        pts2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
        points.extend(pts2)
        
    return np.array(points)

def construct():
    best_score = 0.0
    best_points = None
    
    # We search over topologies (n1, n2) such that 1 + n1 + n2 = 13
    # So n1 + n2 = 12.
    # Possible pairs (n1, n2) where n1, n2 >= 1:
    # (1, 11), (2, 10), ..., (11, 1)
    
    topologies = []
    for n1 in range(1, 12):
        n2 = 12 - n1
        if n2 >= 1:
            topologies.append((n1, n2))
            
    # We will perform a global grid search for each topology.
    # Parameters: r1, r2, phi_rel (phi1 fixed to 0)
    # Range for r: [0.1, 1.0]
    # Range for phi: [0, 2*pi/n2)
    
    # To balance time and accuracy, we use a two-stage approach or a sufficiently fine single stage.
    # Given vectorized scoring is fast, we can do a fairly fine grid.
    
    # Stage 1: Coarse Grid
    # r1, r2: 30 points in [0.1, 1.0]
    # phi: 15 points in [0, 2*pi/n2)
    
    r_coarse = np.linspace(0.1, 1.0, 30)
    
    candidates = []
    
    for n1, n2 in topologies:
        phi_coarse = np.linspace(0.0, 2 * np.pi / n2, 15)
        
        for r1 in r_coarse:
            for r2 in r_coarse:
                for phi2 in phi_coarse:
                    points = _generate_concentric_points(n1, r1, 0.0, n2, r2, phi2, include_center=True)
                    score = _score_configuration(points)
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
                    candidates.append((score, r1, r2, phi2, n1, n2))
    
    # Stage 2: Fine Grid around top candidates
    # Sort candidates
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 50 candidates to refine
    top_candidates = candidates[:50]
    
    # Deduplicate roughly by checking if parameters are close?
    # Or just refine all top 50.
    # Refinement grid:
    # r: 11 points in [r0-0.05, r0+0.05]
    # phi: 11 points in [phi0 - pi/n2, phi0 + pi/n2]
    
    for score, r1_c, r2_c, phi2_c, n1, n2 in top_candidates:
        # Define local bounds
        r1_min = max(0.05, r1_c - 0.05)
        r1_max = min(1.0, r1_c + 0.05)
        r2_min = max(0.05, r2_c - 0.05)
        r2_max = min(1.0, r2_c + 0.05)
        
        # Angular period is 2*pi/n2. We search a window of +/- 1/4 period? Or just +/- small amount.
        # The function is periodic with period 2*pi/n2.
        # We can just search a small window around the coarse best.
        # Window size: 0.5 * (2*pi/n2) ? No, that might miss the peak if coarse grid was off.
        # Let's use a window of +/- 0.1 * (2*pi/n2) ?
        # Actually, just use a fixed small window in radians, e.g., +/- 0.2 rad.
        # Or better, since we know the period, we can map to [0, 2*pi/n2) and search locally.
        
        # Let's just do a fine grid in a box around the coarse point.
        r1_fine = np.linspace(r1_min, r1_max, 11)
        r2_fine = np.linspace(r2_min, r2_max, 11)
        
        # For phi, let's search +/- 0.1 radians? Or +/- 1/20 of the period?
        # Period = 2*pi/n2.
        # Let's search +/- 0.05 * period.
        phi_period = 2 * np.pi / n2
        phi_min = phi2_c - 0.05 * phi_period
        phi_max = phi2_c + 0.05 * phi_period
        phi_fine = np.linspace(phi_min, phi_max, 11)
        
        for r1 in r1_fine:
            for r2 in r2_fine:
                for phi2 in phi_fine:
                    # Ensure phi is within valid range? The generator handles any phi.
                    points = _generate_concentric_points(n1, r1, 0.0, n2, r2, phi2, include_center=True)
                    score = _score_configuration(points)
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()

    # Fallback
    if best_points is None:
        angles = np.arange(13) * (2 * np.pi / 13)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
