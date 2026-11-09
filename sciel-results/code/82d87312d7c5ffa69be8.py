import numpy as np
import time

def _hex_vertices(cx, cy, d, a_deg):
    a_rad = np.deg2rad(a_deg)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_outer_side(positions, angle_deg):
    """
    Compute the minimum side length of a regular hexagon with the given angle
    that encloses all unit hexagons at the given positions.
    """
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    
    # All inner hexagon vertices
    all_verts = positions[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)
    
    a_rad = np.deg2rad(angle_deg)
    
    # Outward normals of the outer hexagon edges
    normals = []
    for i in range(6):
        n_angle = a_rad + i * np.pi/3 + np.pi/6
        n_unit = np.array([np.cos(n_angle), np.sin(n_angle)])
        normals.append(n_unit)
    normals = np.array(normals) # (6, 2)
    
    # Project all vertices onto normals
    projections = all_verts_flat @ normals.T # (36, 6)
    max_projections = np.max(projections, axis=0) # (6,)
    
    # For a regular hexagon of side s, the distance from center to edge (apothem) is s * sqrt(3) / 2.
    # So max_proj <= s * sqrt(3) / 2
    # s >= 2 * max_proj / sqrt(3)
    min_s = np.max(max_projections) * 2 / np.sqrt(3)
    
    return min_s

def _check_validity(inner, outer, s):
    """
    Validates the solution according to the task contract.
    Returns True if valid, False otherwise.
    """
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    
    if s < np.sqrt(12) - 1e-9:
        return False
        
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-9:
                return False
                
    inner_verts_list = []
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1, ang)
        # Check unit edge length
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False
        inner_verts_list.append(verts)

    # Check separation (SAT)
    for i in range(12):
        for j in range(i+1, 12):
            va = inner_verts_list[i]
            vb = inner_verts_list[j]
            
            separated = False
            # Check axes from A
            for k in range(6):
                v1 = va[k]
                v2 = va[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if separated: continue
            
            # Check axes from B
            for k in range(6):
                v1 = vb[k]
                v2 = vb[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if not separated:
                return False
                
    # Check inner vertices inside outer
    for i in range(12):
        for v in inner_verts_list[i]:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                e = p2 - p1
                cross_val = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
                if cross_val < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
                
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580 # Safety margin
    
    # Strategy:
    # The problem asks to pack 12 unit hexagons in a regular hexagon.
    # The most efficient packing of equal circles/hexagons is a hexagonal lattice.
    # For 12 hexagons, the optimal configuration is a central hexagon surrounded by 6,
    # plus 5 more attached to the boundary, or a 3x4 rectangular block, etc.
    # However, the "hexagonal" arrangement of 12 is typically:
    # 1 center, 6 in first ring, 5 in second ring? No, that's 12.
    # Let's check the standard "hex number" arrangements.
    # 1, 7, 19... 12 is not a centered hexagonal number.
    # But 12 can be arranged in a 3x4 grid or other shapes.
    # The most compact arrangement for 12 hexagons is likely the one that minimizes the bounding hexagon.
    # A common compact shape is a 3x4 parallelogram or a hexagon with corners cut.
    # Actually, for 12 hexagons, the optimal packing in a hexagon is known to be a specific lattice arrangement.
    # Let's construct the "hexagonal" lattice positions for 12 hexagons.
    # The centers of a triangular lattice with spacing d (distance between centers of touching hexagons)
    # for unit hexagons (side 1) is d = sqrt(3) * 1 = sqrt(3) if they are touching side-to-side?
    # Wait. Distance between centers of two touching unit hexagons (side 1) is sqrt(3) * 1?
    # Apothem of unit hexagon is sqrt(3)/2.
    # If two hexagons touch side-to-side, the distance between centers is 2 * apothem = sqrt(3).
    # So the lattice spacing is sqrt(3).
    
    # Let's generate the 12 positions on a triangular lattice with spacing d = sqrt(3).
    # We want to center them at (0,0) to minimize the bounding box.
    # The 12 hexagons in the most compact "hexagonal" shape are:
    # Ring 0: (0,0) -> 1
    # Ring 1: 6 neighbors -> 6
    # Ring 2: 5 of the 12 neighbors? 
    # Total 12.
    # Let's list the coordinates in axial coordinates (q, r) for a hex grid.
    # (0,0)
    # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Next ring: (2,-1), (2,-2), (1,-2), (0,-2), (-1,1) is already there? No.
    # Let's just use the standard Cartesian coordinates for a triangular lattice.
    # Basis vectors:
    # v1 = (d, 0)
    # v2 = (d/2, d*sqrt(3)/2)
    # where d = sqrt(3).
    
    d = np.sqrt(3)
    
    # Axial coordinates (q, r) for the 12 hexagons in a compact shape.
    # A "3x4" block is not hexagonal.
    # The optimal shape for 12 is often a hexagon with side lengths 1, 2, 2, 1, 2, 2? No.
    # Let's try the shape: 1 center, 6 around it, and 5 more attached to the "flat" sides of the 6-ring.
    # This forms a hexagon of side 2 with 3 corners missing?
    # Side 2 hexagon has 1 + 6 + 12 = 19 hexagons.
    # If we remove the 7 corners (1 center of each corner triangle?), no.
    # Let's just use the 12 positions that form a 3x4 parallelogram?
    # Or better, let's use the known optimal packing for 12.
    # According to literature, the optimal packing of 12 unit hexagons in a hexagon
    # is a specific lattice arrangement.
    # Let's assume the positions are on a triangular lattice with spacing d=sqrt(3).
    # We will try a few candidate sets of 12 lattice points and pick the one with smallest s.
    
    candidates = []
    
    # Candidate 1: 3x4 grid (parallelogram)
    # Points: (i*v1 + j*v2) for i in 0..2, j in 0..3
    # This is 12 points.
    v1 = np.array([d, 0])
    v2 = np.array([d/2, d*np.sqrt(3)/2])
    
    pts = []
    for i in range(3):
        for j in range(4):
            pts.append(i*v1 + j*v2)
    pts = np.array(pts)
    # Center it
    pts -= np.mean(pts, axis=0)
    candidates.append(pts)
    
    # Candidate 2: Hexagonal shape (1 center, 6 ring, 5 extra)
    # This is harder to define directly.
    # Let's try a "diamond" shape or other compact shapes.
    # Actually, the 3x4 parallelogram is likely not the most compact for a hexagonal container.
    # A more symmetric shape is better.
    # Let's try the 12 points closest to the origin in the triangular lattice.
    # Generate many lattice points, sort by distance to origin, take first 12.
    
    lattice_pts = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            p = i*v1 + j*v2
            lattice_pts.append(p)
    lattice_pts = np.array(lattice_pts)
    dists = np.linalg.norm(lattice_pts, axis=1)
    idx = np.argsort(dists)
    # Take first 12
    pts2 = lattice_pts[idx[:12]]
    # Center it
    pts2 -= np.mean(pts2, axis=0)
    candidates.append(pts2)
    
    # Candidate 3: Another symmetric shape.
    # Maybe a 4x3 grid rotated?
    v1_rot = np.array([d/2, d*np.sqrt(3)/2])
    v2_rot = np.array([-d/2, d*np.sqrt(3)/2])
    pts3 = []
    for i in range(4):
        for j in range(3):
            pts3.append(i*v1_rot + j*v2_rot)
    pts3 = np.array(pts3)
    pts3 -= np.mean(pts3, axis=0)
    candidates.append(pts3)
    
    best_inner = None
    best_outer = None
    best_s = float('inf')
    best_score = 0.0
    
    # We assume the outer hexagon is aligned with the inner lattice (angle 0)
    # or we can optimize the outer angle slightly, but for a lattice packing,
    # the optimal outer hexagon angle is usually 0 or 30 degrees relative to the lattice.
    # Let's try angle 0 and angle 30.
    
    for cand_pts in candidates:
        for outer_angle_deg in [0, 30]:
            s = _min_outer_side(cand_pts, outer_angle_deg)
            
            inner = np.zeros((12, 3))
            inner[:, :2] = cand_pts
            inner[:, 2] = 0.0 # Inner hexagons aligned with 0
            
            outer = np.array([0, 0, outer_angle_deg])
            
            if _check_validity(inner, outer, s):
                score = 1.0 / s
                if score > best_score:
                    best_score = score
                    best_inner = inner
                    best_outer = outer
                    best_s = s
                    
    # If no valid solution found (should not happen), return a fallback
    if best_inner is None:
        # Use the first candidate with a larger s to ensure validity
        pts = candidates[0]
        s = _min_outer_side(pts, 0) * 1.1 # Add margin
        inner = np.zeros((12, 3))
        inner[:, :2] = pts
        inner[:, 2] = 0.0
        outer = np.array([0, 0, 0])
        return inner, outer, s
        
    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END
