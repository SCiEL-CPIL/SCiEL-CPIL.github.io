# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon with center (cx,cy), side d, rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _check_validity(inner, outer, s, tol=1e-8):
    """Validate the packing. Returns True if valid."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if s < np.sqrt(12.0) - 1e-12:
        return False

    # Compute outer vertices
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    if len(np.unique(np.round(outer_verts, 10), axis=0)) < 6:
        return False

    # Check unit edge lengths of inner hexagons
    for i in range(12):
        verts = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for j in range(6):
            v1 = verts[j]
            v2 = verts[(j + 1) % 6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > tol:
                return False

    # Separation check: SAT for each pair of inner hexagons
    for i in range(12):
        vi = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for j in range(i + 1, 12):
            vj = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            # Collect edge normals from both hexagons
            normals = []
            for verts in (vi, vj):
                for k in range(6):
                    e = verts[(k + 1) % 6] - verts[k]
                    n = np.array([-e[1], e[0]])
                    normals.append(n)
            # SAT: if ANY axis separates, they are separated (valid)
            sep = False
            for n in normals:
                proj_i = vi @ n
                proj_j = vj @ n
                min_i, max_i = proj_i.min(), proj_i.max()
                min_j, max_j = proj_j.min(), proj_j.max()
                gap = max(min_i, min_j) - min(max_i, max_j)
                if gap > 1e-6:
                    sep = True
                    break
            if not sep:
                return False

    # Containment check: each inner vertex must be inside outer hexagon
    for i in range(12):
        vi = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for v in vi:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False

    return True


def hexagon_packing_12():
    """
    Packs 12 unit regular hexagons in a regular hexagon using a honeycomb lattice arrangement.
    Returns (inner, outer, s) where inner is (12,3), outer is (3,), s is side length.
    """
    deadline = time.time() + 620  # Safety margin before 630s limit
    
    # Honeycomb lattice: unit hexagons tile edge-to-edge
    # For 12 hexagons, use a compact cluster
    # 
    # Hexagonal lattice basis vectors for hexagons with side 1:
    # Adjacent hexagon centers are at distance sqrt(3) apart
    # 
    # Use axial coordinates (q, r) for hex grid:
    # Position: (sqrt(3)*(q + r/2), (3/2)*r) for pointy-top, or
    # Position: (sqrt(3)*(q + r/2), (3/2)*r) 
    #
    # For flat-top hexagons (angle 0), neighbors are at:
    # (sqrt(3), 0), (sqrt(3)/2, 3/2), (-sqrt(3)/2, 3/2), 
    # (-sqrt(3), 0), (-sqrt(3)/2, -3/2), (sqrt(3)/2, -3/2)
    
    sqrt3 = np.sqrt(3.0)
    
    # Define 12 hexagon positions in a compact cluster
    # Using axial coordinates and converting to Cartesian
    # 
    # A good compact shape: 4-4-4 in three rows, or a hexagon-like shape
    # 
    # Let's use a shape that fits in a hexagon of side 3:
    # Row 0 (top): 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 4 hexagons
    # Row 3 (bottom): 1 hexagon
    # Total: 12
    #
    # Actually, let's use the standard "12-hexagon" compact shape
    # that is known to fit in a hexagon of side 3.
    
    # Positions in terms of hex grid coordinates
    # Using offset coordinates for flat-top hexagons
    # 
    # For flat-top hexagons with side 1:
    # x = sqrt(3) * (col + row/2)
    # y = 1.5 * row
    
    positions = []
    
    # Compact 12-hexagon arrangement:
    # Think of it as a larger hexagon with side 2 (which has 7 hexagons) 
    # plus 5 more, or use a rectangular-ish patch.
    #
    # Let's use this arrangement (row, col) in offset coords:
    # Row 0: cols 0, 1, 2
    # Row 1: cols 0, 1, 2, 3
    # Row 2: cols 0, 1, 2, 3
    # Row 3: col 1
    # Total: 3+4+4+1 = 12
    
    hex_positions = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2), (2, 3),
        (3, 1),
    ]
    
    for row, col in hex_positions:
        x = sqrt3 * (col + row / 2.0)
        y = 1.5 * row
        positions.append([x, y])
    
    positions = np.array(positions)
    
    # Center the configuration
    centroid = positions.mean(axis=0)
    positions = positions - centroid
    
    # Compute the minimal enclosing regular hexagon
    # Try different outer hexagon orientations and find the smallest s
    
    best_s = np.inf
    best_inner = None
    best_outer = None
    
    # All inner hexagons have angle 0 (flat-top)
    inner_angle = 0.0
    
    for outer_rot_deg in range(0, 60, 1):
        if time.time() > deadline:
            break
        
        outer_angle = np.radians(outer_rot_deg)
        
        # The 6 outward normal directions of the outer hexagon
        # For a regular hexagon with angle a, the edges are perpendicular to directions a + 30 + 60k
        # Actually, the vertices are at angles a + 60k, so edges are between them
        # The normal to each edge points outward
        # 
        # For a hexagon with vertices at angles a, a+60, ..., the edge from vertex i to i+1
        # has a normal direction. The inradius direction (center to edge midpoint) is at angle a + 30 + 60k
        
        normal_angles = outer_angle + np.radians(30.0) + np.radians(60.0 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Compute all inner hexagon vertices
        all_verts = []
        for pos in positions:
            verts = _hex_vertices(pos[0], pos[1], 1.0, inner_angle)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # The outer hexagon centered at origin with side s has inradius = s * sqrt(3)/2
        # For containment: for each normal direction n_k, max(v . n_k) <= s * sqrt(3)/2
        supports = all_verts @ normals.T  # shape (72, 6)
        max_supports = supports.max(axis=0)  # shape (6,)
        
        # Also need to check the negative directions? No, the 6 normals already cover all directions
        # since they are 60 degrees apart and we take max over all vertices
        
        required_s = np.max(max_supports) / (sqrt3 / 2.0)
        
        if required_s < best_s - 1e-10:
            best_s = required_s
            
            inner = np.zeros((12, 3))
            for i, pos in enumerate(positions):
                inner[i, 0] = pos[0]
                inner[i, 1] = pos[1]
                inner[i, 2] = inner_angle
            
            outer = np.array([0.0, 0.0, float(outer_rot_deg)])
            best_inner = inner
            best_outer = outer
    
    if best_inner is None:
        # Fallback
        inner = np.zeros((12, 3))
        for i, pos in enumerate(positions):
            inner[i, 0] = pos[0]
            inner[i, 1] = pos[1]
            inner[i, 2] = 0.0
        outer = np.array([0.0, 0.0, 0.0])
        s = 3.0
        return inner, outer, s
    
    # Final validation
    if not _check_validity(best_inner, best_outer, best_s):
        # Fallback with larger s
        inner = np.zeros((12, 3))
        for i, pos in enumerate(positions):
            inner[i, 0] = pos[0]
            inner[i, 1] = pos[1]
            inner[i, 2] = 0.0
        outer = np.array([0.0, 0.0, 0.0])
        s = 3.5
        return inner, outer, s
    
    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
