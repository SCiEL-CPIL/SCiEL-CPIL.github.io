import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


def _check_validity(inner, outer, s):
    """Validate packing; return (is_valid, error_msg)."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False, "non-finite values"
    if s < np.sqrt(12.0) - 1e-12:
        return False, "s < sqrt(12)"
    
    # Outer hexagon vertices must be distinct
    ov = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(ov[i] - ov[j]) < 1e-10:
                return False, "outer vertices not distinct"
    
    # Check unit edge length for inner hexagons
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for i in range(6):
            edge_len = np.linalg.norm(hv[(i + 1) % 6] - hv[i])
            if abs(edge_len - 1.0) > 1e-8:
                return False, f"unit edge length differs: {edge_len}"
    
    # Check inner-inner separation using SAT
    def _separated(v1, v2):
        """Check if two hexagons are separated (non-overlapping) using SAT."""
        # Get all edges from both hexagons
        edges = []
        for i in range(6):
            edges.append(v1[(i + 1) % 6] - v1[i])
            edges.append(v2[(i + 1) % 6] - v2[i])
        
        for edge in edges:
            # Normal direction
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-12:
                continue
            normal /= np.linalg.norm(normal)
            
            # Project both hexagons onto normal
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check if intervals are separated by more than 1e-6
            gap1 = min2 - max1  # v2 to the right of v1
            gap2 = min1 - max2  # v1 to the right of v2
            if gap1 > 1e-6 or gap2 > 1e-6:
                return True  # Found a separating axis
        
        return False  # No separating axis found -> overlapping
    
    inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner]
    for i in range(12):
        for j in range(i + 1, 12):
            if not _separated(inner_verts[i], inner_verts[j]):
                return False, f"inner hexagons {i} and {j} overlap"
    
    # Check inner vertices inside outer hexagon
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for v in hv:
            # For every outer edge (p1, p2) in vertex order, cross(p2-p1, v-p1) >= -1e-6
            for k in range(6):
                p1 = ov[k]
                p2 = ov[(k + 1) % 6]
                edge = p2 - p1
                to_v = v - p1
                cross = edge[0] * to_v[1] - edge[1] * to_v[0]
                if cross < -1e-6:
                    return False, f"inner vertex outside outer hexagon"
    
    return True, ""


def _compute_min_s(inner):
    """
    Compute the minimum outer hexagon side length needed to contain all inner hexagons.
    Assumes outer hexagon is centered at origin with angle 0.
    """
    # For a regular hexagon with side s centered at origin, angle 0:
    # Vertices are at (s*cos(60*i), s*sin(60*i)) for i=0..5
    # The hexagon contains points where max over 6 directions of |dot(point, dir)| <= s * cos(30)
    # Actually, for a regular hexagon, the distance from center to each side is s * sqrt(3)/2
    # And the direction to each side is at angles 30, 90, 150, 210, 270, 330 degrees
    
    # The 6 outward normals to the sides are at angles 30, 90, 150, 210, 270, 330
    normal_angles = np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # For each inner hexagon, compute the maximum projection onto each normal
    max_projections = np.zeros(6)
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        # Project all vertices onto each normal
        projections = hv @ normals.T  # shape (6, 6)
        max_projections = np.maximum(max_projections, np.max(projections, axis=0))
    
    # The minimum s such that s * sqrt(3)/2 >= max_projection for all directions
    # So s >= max_projection / (sqrt(3)/2) = 2 * max_projection / sqrt(3)
    s_min = np.max(max_projections) / (np.sqrt(3.0) / 2.0)
    
    # Add a small margin to ensure validity
    return s_min + 1e-4


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s limit
    
    # Known good configuration: 12 hexagons in a hexagonal pattern
    # Center + 6 around + 6 more
    # For unit hexagons, the distance between centers of adjacent hexagons is 2*sqrt(3)/2 = sqrt(3) ≈ 1.732
    # Wait, let me recalculate. For two adjacent unit hexagons sharing an edge,
    # the distance between centers is 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732
    
    # Actually, for regular hexagons with side 1, the distance between centers of
    # two hexagons that share an edge is 2 * (sqrt(3)/2) = sqrt(3)
    
    # Let's use a known optimal configuration
    # 12 hexagons: 1 center + 6 ring + 6 more
    
    # The optimal packing for 12 unit hexagons in a hexagon
    # Based on literature, the minimal containing hexagon has side length around 4.5-5
    
    # Let's start with a symmetric arrangement
    # Hexagonal lattice spacing: sqrt(3) for adjacent hexagons
    
    d = np.sqrt(3.0)  # Distance between centers of adjacent hexagons
    
    # 12 hexagons in a pattern:
    # Row 0 (y=0): x = -3d/2, -d/2, d/2, 3d/2 (4 hexagons)
    # Row 1 (y=d*sqrt(3)/2): x = -d, 0, d (3 hexagons)  
    # Row 2 (y=-d*sqrt(3)/2): x = -d, 0, d (3 hexagons)
    # That's 10. Need 2 more.
    
    # Alternative: 
    # Let me use the configuration from the seed but verify it's valid
    
    inner = np.array([
        [0.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [-2.5, 0.0, 0.0],
        [1.25, 2.1650635, 0.0],
        [-1.25, 2.1650635, 0.0],
        [1.25, -2.1650635, 0.0],
        [-1.25, -2.1650635, 0.0],
        [3.75, 2.1650635, 0.0],
        [-3.75, 2.1650635, 0.0],
        [3.75, -2.1650635, 0.0],
        [-3.75, -2.1650635, 0.0],
        [0.0, -4.330127, 0.0],
    ])
    outer = np.array([0.0, 0.0, 0.0])
    
    # Compute minimum s needed
    s = _compute_min_s(inner)
    
    # Validate initial guess
    valid, msg = _check_validity(inner, outer, s)
    if not valid:
        print(f"Initial guess invalid: {msg}")
        # Try a larger s
        s = s + 1.0
        valid, msg = _check_validity(inner, outer, s)
        if not valid:
            print(f"Still invalid: {msg}")
            # Fallback: just return a very large outer hex
            outer_hex_data = np.array([0.0, 0.0, 0.0])
            s = 20.0
            return inner, outer_hex_data, s
    
    best_s = s
    best_inner = inner.copy()
    best_outer = outer.copy()
    
    # Try to optimize using scipy - optimize positions to minimize s
    try:
        def obj_func(params):
            """
            params: 36 values: 12*(x,y) - we fix angles to 0 for simplicity
            Returns: minimum s needed + penalty for overlaps
            """
            inner_p = params.reshape(12, 2)
            # Add angle 0
            inner_full = np.column_stack([inner_p, np.zeros(12)])
            
            # Check for overlaps using center distances
            # For two unit hexagons with same orientation, they don't overlap if
            # center distance > 2 * sqrt(3)/2 = sqrt(3)
            # Actually, the minimum distance between centers of non-overlapping
            # unit hexagons (same orientation) is 2 * (sqrt(3)/2) = sqrt(3)
            
            penalty = 0.0
            for i in range(12):
                for j in range(i + 1, 12):
                    dist = np.linalg.norm(inner_p[i] - inner_p[j])
                    min_dist = np.sqrt(3.0)  # Minimum center distance for non-overlap
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2
            
            # Compute minimum s
            s_min = _compute_min_s(inner_full)
            
            return s_min + penalty
        
        x0 = inner[:, :2].flatten()
        
        result = minimize(obj_func, x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7})
        
        if result.fun < best_s:
            inner_opt_pos = result.x.reshape(12, 2)
            inner_opt = np.column_stack([inner_opt_pos, np.zeros(12)])
            s_opt = _compute_min_s(inner_opt)
            outer_opt = np.array([0.0, 0.0, 0.0])
            
            # Validate
            valid, msg = _check_validity(inner_opt, outer_opt, s_opt)
            if valid:
                best_inner = inner_opt
                best_outer = outer_opt
                best_s = s_opt
                print(f"Optimized: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
            else:
                print(f"Optimized result invalid: {msg}")
        
        if time.time() > deadline:
            return best_inner, best_outer, best_s
        
        # Try more iterations if time permits
        if time.time() < deadline - 60:
            result2 = minimize(obj_func, result.x, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result2.fun < best_s:
                inner_opt_pos = result2.x.reshape(12, 2)
                inner_opt = np.column_stack([inner_opt_pos, np.zeros(12)])
                s_opt = _compute_min_s(inner_opt)
                outer_opt = np.array([0.0, 0.0, 0.0])
                
                valid, msg = _check_validity(inner_opt, outer_opt, s_opt)
                if valid:
                    best_inner = inner_opt
                    best_outer = outer_opt
                    best_s = s_opt
                    print(f"Further optimized: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
    
    except Exception as e:
        print(f"Optimization error: {e}")
    
    # Final validation
    valid, msg = _check_validity(best_inner, best_outer, best_s)
    if not valid:
        print(f"Final result invalid: {msg}, falling back to safe configuration")
        # Safe fallback
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        s = 20.0
        return inner, outer_hex_data, s
    
    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
