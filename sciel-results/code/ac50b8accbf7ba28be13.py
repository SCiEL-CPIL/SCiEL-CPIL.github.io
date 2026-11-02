import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13

def _score_configuration(points):
    """
    Calculate the score for a given set of points.
    Score = min_triangle_area / convex_hull_area
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
    # For n=13, there are 286 triangles.
    # We can compute all areas at once using broadcasting or explicit loops.
    # Explicit loop is fine for n=13, but let's try to make it fast.
    
    min_area = float('inf')
    
    # Pre-compute differences? 
    # area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    
    # Using itertools.combinations is clean
    from itertools import combinations
    idx = range(n)
    for i, j, k in combinations(idx, 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        
        # Cross product magnitude
        dx1 = p2[0] - p1[0]
        dy1 = p2[1] - p1[1]
        dx2 = p3[0] - p1[0]
        dy2 = p3[1] - p1[1]
        
        area = 0.5 * abs(dx1 * dy2 - dx2 * dy1)
        if area < min_area:
            min_area = area
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
    # Note: (n1, n2) with phi1, phi2 is equivalent to (n2, n1) with swapped radii/angles if we allow independent angles?
    # Actually, the ring at radius r1 with n1 points is distinct from ring at r2 with n2 points.
    # However, due to rotational symmetry of the whole system, we can fix the global rotation.
    # Let's fix phi1 = 0 for the first ring to reduce dimensionality? 
    # No, the relative angle phi_rel = phi2 - phi1 matters.
    # We can fix phi1 = 0.0 and search phi2 in [0, 2*pi/n2).
    
    topologies = []
    for n1 in range(1, 12):
        n2 = 12 - n1
        if n2 >= 1:
            topologies.append((n1, n2))
    
    # To avoid duplicate work, note that (n1, n2) and (n2, n1) are structurally similar 
    # if we swap the roles of the rings. 
    # However, r1 and r2 are distinct variables. 
    # (n1=1, n2=11) means 1 point on ring 1, 11 on ring 2.
    # (n1=11, n2=1) means 11 points on ring 1, 1 on ring 2.
    # These are different geometric configurations unless we swap r1<->r2 and n1<->n2.
    # Since we search r1, r2 fully, we should probably just run all pairs.
    # But (n1, n2) and (n2, n1) cover the same set of point sets if we allow r1>r2 or r2>r1?
    # Yes, the set of points {center} U Ring(n1, r1, 0) U Ring(n2, r2, phi) 
    # is the same as {center} U Ring(n2, r2, phi) U Ring(n1, r1, 0) 
    # if we just rename the rings.
    # So we can restrict to n1 <= n2 to halve the search?
    # Let's check: 
    # Config A: 1 point at r1, 11 points at r2.
    # Config B: 11 points at r1, 1 point at r2.
    # If in Config A, r1 < r2, we have 1 inner, 11 outer.
    # If in Config B, r1 < r2, we have 11 inner, 1 outer.
    # These are distinct.
    # So we must search all (n1, n2).
    
    candidates = []
    
    for n1, n2 in topologies:
        # Grid search parameters
        # r1, r2 in [0.1, 1.0]
        # phi2 in [0, 2*pi/n2) (phi1 fixed at 0)
        
        # Coarse grid
        r_vals = np.linspace(0.1, 1.0, 15)
        phi_vals = np.linspace(0.0, 2 * np.pi / n2, 15)
        
        for r1 in r_vals:
            for r2 in r_vals:
                # Skip if radii are too close to cause degeneracy? 
                # Or just let the score handle it.
                for phi2 in phi_vals:
                    points = _generate_concentric_points(n1, r1, 0.0, n2, r2, phi2, include_center=True)
                    score = _score_configuration(points)
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
                    
                    # Store top candidates for local optimization
                    if score > 0.01: # Only store non-trivial scores
                        candidates.append((score, [r1, r2, phi2], n1, n2))
                        
    # Local optimization on the best few candidates
    # Sort candidates by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 20 unique-ish candidates
    # We need to be careful not to optimize the same point multiple times.
    # Simple deduplication: just take the top 10 from the coarse grid.
    
    top_candidates = candidates[:10]
    
    for score, params, n1, n2 in top_candidates:
        # params: [r1, r2, phi2]
        # Objective: maximize score => minimize -score
        
        def objective(params):
            r1, r2, phi2 = params
            # Bound check
            if r1 < 0.01 or r2 < 0.01:
                return 1.0 # Large penalty (minimizing -score, so large positive is bad)
            points = _generate_concentric_points(n1, r1, 0.0, n2, r2, phi2, include_center=True)
            s = _score_configuration(points)
            return -s
        
        x0 = np.array(params)
        bounds = [(0.01, 1.0), (0.01, 1.0), (0.0, 2 * np.pi / n2)]
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success:
                r1_opt, r2_opt, phi2_opt = res.x
                points_opt = _generate_concentric_points(n1, r1_opt, 0.0, n2, r2_opt, phi2_opt, include_center=True)
                score_opt = _score_configuration(points_opt)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_points = points_opt.copy()
                    
        except Exception:
            pass

    # Fallback
    if best_points is None:
        angles = np.arange(13) * (2 * np.pi / 13)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
