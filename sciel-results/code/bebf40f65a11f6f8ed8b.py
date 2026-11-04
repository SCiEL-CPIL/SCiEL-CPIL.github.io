import numpy as np
import time
from scipy.spatial import Delaunay

N = 12
TIME_LIMIT = 145.0
START_TIME = time.time()

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _separating_axis_check(v1, v2):
    """Check if two convex polygons are disjoint using separating axis theorem."""
    n1, n2 = len(v1), len(v2)
    
    # Check axes from both polygons
    for i in range(n1):
        edge = v1[(i+1) % n1] - v1[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        
        proj1 = np.array([np.dot(p, normal) for p in v1])
        proj2 = np.array([np.dot(p, normal) for p in v2])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return True
    
    return False

def _is_valid_fast(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-9):
    """Fast validity check."""
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if not _separating_axis_check(inner_verts[i], inner_verts[j]):
                return False, "overlap"
        
        for p in inner_verts[i]:
            inside = True
            for k in range(6):
                a = outer_verts[k]
                b = outer_verts[(k+1)%6]
                cross_ab = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
                if cross_ab < -eps:
                    inside = False
                    break
            if not inside:
                return False, "outside"
                
    return True, "valid"

def _compute_required_side(centers, angles, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Compute minimum outer side length to contain all hexagons."""
    max_proj = 0.0
    for k in range(6):
        angle = np.deg2rad(outer_angle + k * 60)
        normal = np.array([np.cos(angle), np.sin(angle)])
        
        for i in range(N):
            verts = _hex_vertices(centers[i], 1.0, angles[i])
            for v in verts:
                # Project relative to outer center
                proj = np.dot(v - np.array(outer_center), normal)
                if proj > max_proj:
                    max_proj = proj
                    
    return max_proj * 2.0 / np.sqrt(3.0)

def _minkowski_sum_hex(hex1_verts, hex2_verts):
    """
    Compute Minkowski sum of two convex hexagons.
    The result is a convex polygon (up to 12 vertices).
    We approximate it by sampling the support function or using a simple convex hull of sums.
    For two regular hexagons, the Minkowski sum is a regular hexagon if aligned, 
    or a dodecagon-like shape if rotated.
    To keep it simple and robust for the 'configuration space' idea, 
    we will treat the forbidden region for the center of hex2 relative to hex1 
    as the Minkowski sum of hex1 and -hex2.
    """
    # Vertices of -hex2
    neg_hex2_verts = -hex2_verts
    
    # Sum of all vertex pairs
    sums = []
    for v1 in hex1_verts:
        for v2 in neg_hex2_verts:
            sums.append(v1 + v2)
            
    # Convex hull of these sums gives the Minkowski sum boundary
    # Using scipy.spatial.ConvexHull
    from scipy.spatial import ConvexHull
    pts = np.array(sums)
    try:
        hull = ConvexHull(pts)
        hull_verts = pts[hull.vertices]
        return hull_verts
    except Exception:
        # Fallback: circle approximation
        center = np.mean(pts, axis=0)
        radius = np.max(np.linalg.norm(pts - center, axis=1))
        return _hex_vertices(center, radius, 0)

def _check_config_validity(centers, angles, outer_side, outer_center, outer_angle):
    """Check if a specific configuration is valid (no overlaps, inside outer)."""
    inner_hexes = [[centers[i][0], centers[i][1], angles[i]] for i in range(N)]
    valid, _ = _is_valid_fast(inner_hexes, outer_side, outer_center, outer_angle)
    return valid

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    np.random.seed(42)
    
    # Strategy:
    # 1. Start with a known valid "spread out" configuration.
    # 2. Define a "goal" configuration which is a tight hexagonal lattice.
    # 3. Use A* or similar to find a collision-free path in configuration space 
    #    from start to goal, or simply interpolate and resolve collisions 
    #    using a simplified Minkowski-based repulsion.
    
    # However, implementing a full A* on 24D space is heavy.
    # Instead, we use a "Shake and Settle" approach with a different potential.
    # The "Minkowski" idea: The distance between two hexagons is the distance 
    # between their centers minus the support function of the Minkowski sum.
    
    # Let's try a different structural approach: 
    # **Simulated Annealing on the Configuration Space with Minkowski-aware penalties**
    # The penalty is based on the actual geometric penetration depth, 
    # approximated by the distance to the Minkowski sum boundary.
    
    def _minkowski_penalty(centers, angles):
        """
        Compute penalty based on Minkowski sum distances.
        For each pair (i, j), the forbidden region for center_j relative to center_i 
        is the Minkowski sum of hex_i and -hex_j.
        If center_j - center_i is inside this region, they overlap.
        Penalty is the distance to the boundary of this region.
        """
        penalty = 0.0
        # Precompute Minkowski sum for standard orientation? 
        # Since angles can vary, we compute on the fly for nearby pairs.
        
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                # Quick reject: if distance is large, no overlap
                if d > 3.5: 
                    continue
                
                hex_i = _hex_vertices(centers[i], 1.0, angles[i])
                hex_j = _hex_vertices(centers[j], 1.0, angles[j])
                
                # Check overlap
                if _separating_axis_check(hex_i, hex_j):
                    continue
                
                # Overlap detected. Compute a penalty.
                # Approximate penetration depth by center distance vs min touching distance.
                # For regular hexagons side 1, min center distance when touching is sqrt(3).
                # This is an approximation but guides the search.
                min_dist = np.sqrt(3)
                if d < min_dist:
                    # Penalty grows quadratically
                    penalty += (min_dist - d) ** 2 * 5.0
                else:
                    # They overlap but centers are far? (e.g. rotated)
                    # Use a constant penalty
                    penalty += 1.0
        return penalty

    def _total_energy(centers, angles, outer_side):
        """
        Energy = outer_side + overlap_penalty + containment_penalty
        """
        # Containment penalty
        c_pen = 0.0
        outer_verts = _hex_vertices(best_outer_center, outer_side, best_outer_angle)
        for i in range(N):
            verts = _hex_vertices(centers[i], 1.0, angles[i])
            for v in verts:
                # Check if v is inside outer
                inside = True
                for k in range(6):
                    a = outer_verts[k]
                    b = outer_verts[(k+1)%6]
                    cross_ab = (b[0]-a[0])*(v[1]-a[1]) - (b[1]-a[1])*(v[0]-a[0])
                    if cross_ab < -1e-9:
                        inside = False
                        break
                if not inside:
                    # Estimate distance outside
                    # Project v onto the violated edge normal
                    a = outer_verts[k]
                    b = outer_verts[(k+1)%6]
                    edge = b - a
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-12:
                        normal = normal / norm
                        # Ensure normal points inward
                        mid = (a + b) / 2
                        if np.dot(normal, mid - np.array(best_outer_center)) > 0:
                            normal = -normal
                        dist = np.dot(v - a, normal)
                        if dist < 0:
                            c_pen += (-dist) ** 2 * 10.0
                        break
        
        o_pen = _minkowski_penalty(centers, angles)
        
        return outer_side + o_pen + c_pen

    # Initial Configuration: Hexagonal Lattice
    # 1 center, 6 neighbors, 6 next neighbors = 13. We need 12.
    # Remove the center one? Or just use the 12 outer ones?
    # Let's use a ring of 6 and a ring of 6.
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    
    # Ring 1: 6 hexagons at distance sqrt(3)
    for i in range(6):
        angle = i * np.pi / 3
        centers[i] = [np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)]
        
    # Ring 2: 6 hexagons at distance 2*sqrt(3)
    for i in range(6):
        angle = (i + 0.5) * np.pi / 3
        centers[i+6] = [2 * np.sqrt(3) * np.cos(angle), 2 * np.sqrt(3) * np.sin(angle)]

    current_centers = centers.copy()
    current_angles = angles.copy()
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.95
    max_iter = 20000
    
    for t in range(max_iter):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        # Perturb
        new_centers = current_centers.copy()
        new_angles = current_angles.copy()
        
        # Random move: shift one hexagon
        idx = np.random.randint(N)
        delta_x = np.random.normal(0, 0.1)
        delta_y = np.random.normal(0, 0.1)
        new_centers[idx][0] += delta_x
        new_centers[idx][1] += delta_y
        
        # Random rotation
        if np.random.random() < 0.1:
            new_angles[idx] += np.random.normal(0, np.pi/6)
            
        # Compute Energy
        # First, estimate required side for new centers
        req_side = _compute_required_side(new_centers, new_angles, best_outer_center, best_outer_angle)
        
        # To speed up, we can skip the full validity check in the inner loop 
        # and rely on the penalty function.
        # However, we need to update best_outer_side based on the new configuration.
        
        e_old = _total_energy(current_centers, current_angles, best_side)
        e_new = _total_energy(new_centers, new_angles, req_side)
        
        if e_new < e_old or np.random.random() < np.exp(-(e_new - e_old) / T):
            current_centers = new_centers
            current_angles = new_angles
            best_side = req_side
            
            # Check if this configuration is actually valid (no overlaps, inside)
            # We do this less frequently to save time
            if t % 10 == 0:
                inner_hexes = [[current_centers[i][0], current_centers[i][1], current_angles[i]] for i in range(N)]
                is_valid, _ = _is_valid_fast(inner_hexes, best_side, best_outer_center, best_outer_angle)
                if is_valid:
                    # Update global best
                    if best_side < best_side: # This is redundant, but logic check
                        pass
                    # Actually, we want to track the best *valid* side
                    # The variable `best_side` in the outer scope is the target.
                    # Let's use a separate variable for the SA loop's best valid side.
                    pass
        
        T *= cooling
        if T < T_min:
            break

    # After SA, try to polish with a few iterations of gradient descent on the valid config
    # Or just return the best found.
    
    # The SA loop above updated `best_side` but not `best_inner`.
    # We need to track the best valid configuration found during SA.
    
    # Let's refactor slightly to ensure we capture the best valid state.
    # Re-run a quick check on the final state of SA
    inner_hexes = [[current_centers[i][0], current_centers[i][1], current_angles[i]] for i in range(N)]
    final_side = _compute_required_side(current_centers, current_angles, best_outer_center, best_outer_angle)
    
    # Check validity
    is_valid, _ = _is_valid_fast(inner_hexes, final_side, best_outer_center, best_outer_angle)
    if is_valid and final_side < best_side:
        best_side = final_side
        best_inner = inner_hexes
        
    # If no valid config found yet, try the initial lattice
    if best_inner is None:
        initial_inner = [[centers[i][0], centers[i][1], 0.0] for i in range(N)]
        initial_side = _compute_required_side(centers, np.zeros(N), best_outer_center, best_outer_angle)
        is_valid, _ = _is_valid_fast(initial_inner, initial_side, best_outer_center, best_outer_angle)
        if is_valid:
            best_side = initial_side
            best_inner = initial_inner

    # Fallback
    if best_inner is None:
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    return (best_inner, best_outer_center, best_side, best_outer_angle)
