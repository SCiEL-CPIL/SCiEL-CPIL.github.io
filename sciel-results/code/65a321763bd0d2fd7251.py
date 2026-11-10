import numpy as np
import time

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
        edges = []
        for i in range(6):
            edges.append(v1[(i + 1) % 6] - v1[i])
            edges.append(v2[(i + 1) % 6] - v2[i])
        
        for edge in edges:
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-12:
                continue
            normal /= np.linalg.norm(normal)
            
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap1 = min2 - max1
            gap2 = min1 - max2
            if gap1 > 1e-6 or gap2 > 1e-6:
                return True
        
        return False
    
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
            for k in range(6):
                p1 = ov[k]
                p2 = ov[(k + 1) % 6]
                edge = p2 - p1
                to_v = v - p1
                cross = edge[0] * to_v[1] - edge[1] * to_v[0]
                if cross < -1e-6:
                    return False, f"inner vertex outside outer hexagon"
    
    return True, ""

def _min_containing_hex_side(inner, outer_center, outer_angle):
    """
    Compute the minimum side length s of a regular hexagon with given center and angle
    that contains all inner hexagons.
    """
    a = np.deg2rad(outer_angle)
    normal_angles = a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    max_proj = 0.0
    for row in inner:
        cx, cy, ang = row
        hv = _hex_vertices(cx, cy, 1.0, ang)
        # Project vertices relative to outer center
        rel = hv - outer_center
        projections = rel @ normals.T
        max_proj = max(max_proj, np.max(projections))
    
    # s * sqrt(3)/2 >= max_proj => s >= max_proj / (sqrt(3)/2)
    s_min = max_proj / (np.sqrt(3.0) / 2.0)
    return s_min

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Hexagonal lattice basis
    b1 = np.array([np.sqrt(3.0), 0.0])
    b2 = np.array([np.sqrt(3.0) / 2.0, 1.5])
    
    # 12 hexagons in a compact arrangement
    # Center + 6 ring + 5 from next ring
    lattice_coords = np.array([
        [0, 0],
        [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
        [2, 0], [1, 1], [-1, 2], [-2, 1], [-2, 0]
    ], dtype=float)
    
    # Convert to Cartesian
    positions = np.column_stack((
        lattice_coords[:, 0] * b1[0] + lattice_coords[:, 1] * b2[0],
        lattice_coords[:, 0] * b1[1] + lattice_coords[:, 1] * b2[1]
    ))
    
    # Center the configuration
    centroid = np.mean(positions, axis=0)
    positions = positions - centroid
    
    # Scale up slightly to ensure strict separation (gap > 1e-6)
    # The minimum distance between centers is sqrt(3) ~ 1.732.
    # The "width" of a hexagon along the axis to a neighbor is sqrt(3).
    # So they are tangent. We need to push them apart.
    # Scaling positions by (1 + epsilon) increases distance by epsilon * dist.
    # Gap increases by 2 * epsilon * dist / 2 ? No.
    # If centers move from d to d(1+eps), the gap between edges is d(1+eps) - d = d*eps.
    # We need gap > 1e-6. d ~ 1.73. So eps > 1e-6 / 1.73 ~ 5e-7.
    # Let's use eps = 1e-4 to be safe.
    scale_factor = 1.0 + 1e-4
    positions_scaled = positions * scale_factor
    
    best_s = np.inf
    best_inner = None
    best_outer = None
    
    # Grid search over rotation and outer angle
    # Step 1 degree is usually sufficient for this kind of problem.
    # 360 * 360 = 129,600 iterations.
    # Each iteration is cheap.
    
    for rot_deg in range(0, 360, 1):
        if time.time() > deadline:
            break
            
        rot_rad = np.deg2rad(rot_deg)
        R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                      [np.sin(rot_rad), np.cos(rot_rad)]])
        
        rotated_pos = positions_scaled @ R.T
        inner_rot = np.column_stack([rotated_pos, np.zeros(12)])
        
        for out_ang_deg in range(0, 360, 1):
            if time.time() > deadline:
                break
                
            s = _min_containing_hex_side(inner_rot, np.array([0.0, 0.0]), float(out_ang_deg))
            
            if s < best_s:
                best_s = s
                best_inner = inner_rot.copy()
                best_outer = np.array([0.0, 0.0, float(out_ang_deg)])
                
    # Add a small margin to s to ensure containment validity (cross product >= -1e-6)
    # The calculation of s_min is exact for the vertices. 
    # However, floating point errors in _check_validity might complain.
    # Adding 1e-5 is safe.
    if best_inner is not None:
        final_s = best_s + 1e-5
        valid, msg = _check_validity(best_inner, best_outer, final_s)
        if not valid:
            # If it fails, increase s further
            final_s = best_s + 0.1
            valid, msg = _check_validity(best_inner, best_outer, final_s)
            if not valid:
                # Emergency fallback
                final_s = 20.0
        return best_inner, best_outer, final_s
    else:
        # Fallback if no result found (should not happen)
        s = _min_containing_hex_side(np.column_stack([positions_scaled, np.zeros(12)]), np.array([0.0, 0.0]), 0.0) + 0.1
        return np.column_stack([positions_scaled, np.zeros(12)]), np.array([0.0, 0.0, 0.0]), s
