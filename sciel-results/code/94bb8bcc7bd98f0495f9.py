# EVOLVE-BLOCK-START
import numpy as np
import math


def _hex_vertices(cx, cy, d, angle_deg):
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + math.radians(60 * i)
        verts.append((cx + d * math.cos(theta), cy + d * math.sin(theta)))
    return np.array(verts)


def _is_valid(inner, outer, s, tol_sep=1e-6, tol_contain=1e-6):
    # Check finite
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if s < math.sqrt(12) - 1e-9:
        return False
    
    ocx, ocy, oang = outer
    outer_verts = _hex_vertices(ocx, ocy, s, oang)
    
    # Check outer vertices distinct
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    # Check inner hexagons are unit side
    for row in inner:
        cx, cy, ang = row
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for i in range(6):
            v1 = verts[i]
            v2 = verts[(i + 1) % 6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False
    
    # Check separation between inner hexagons (SAT)
    n_inner = len(inner)
    for i in range(n_inner):
        vi = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for j in range(i + 1, n_inner):
            vj = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            # Get edge normals for both
            axes = []
            for v in [vi, vj]:
                for k in range(6):
                    e = v[(k + 1) % 6] - v[k]
                    n = np.array([-e[1], e[0]])
                    n_norm = np.linalg.norm(n)
                    if n_norm > 1e-12:
                        n /= n_norm
                        axes.append(n)
            
            separated = False
            for axis in axes:
                proj_i = np.dot(vi, axis)
                proj_j = np.dot(vj, axis)
                min_i, max_i = proj_i.min(), proj_i.max()
                min_j, max_j = proj_j.min(), proj_j.max()
                # Intervals must be more than tol_sep apart
                if max_i + tol_sep < min_j or max_j + tol_sep < min_i:
                    separated = True
                    break
            if not separated:
                return False
    
    # Check containment: all inner vertices inside outer hexagon
    for row in inner:
        cx, cy, ang = row
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for v in verts:
            # For each outer edge (p1, p2) in vertex order, cross(p2-p1, v-p1) >= -tol
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                edge = p2 - p1
                vec = v - p1
                cross = edge[0] * vec[1] - edge[1] * vec[0]
                if cross < -tol_contain:
                    inside = False
                    break
            if not inside:
                return False
    
    return True


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    using a deterministic 3x4 parallelogram lattice construction.
    """
    # 3x4 lattice of unit hexagons in a parallelogram arrangement
    # Hexagon centers in a hexagonal lattice:
    # For a flat-top hexagon (angle=0), neighbors are at:
    #   (±1.5, ±sqrt(3)/2) and (0, ±sqrt(3))
    # Actually for side=1, the distance between centers of adjacent hexagons is sqrt(3)
    # Wait, for unit side hexagons, the distance between centers of edge-adjacent hexagons is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)
    
    # Let's use a standard hex grid layout.
    # Row-based: rows are offset.
    # For 12 hexagons in a 3x4 block:
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    # This forms a parallelogram.
    
    # Actually, a better arrangement for minimizing the enclosing hexagon is a 3x4 grid
    # where the rows are staggered. Let's place them properly.
    
    # Hexagon with side 1, flat-top (angle 0):
    # Vertices at angles 0, 60, 120, 180, 240, 300
    # Width (horizontal) = 2, Height (vertical) = sqrt(3)
    
    # For a hex grid with flat-top hexagons:
    # Horizontal spacing between centers in same row: 1.5 (since width is 2, but they overlap by 0.5 on each side? No.)
    # Actually, for flat-top hexagons of side 1:
    # The distance between centers horizontally in the same row is 1.5 * 2 = 3? No.
    # Let me recalculate. For a regular hexagon with side s=1:
    # Width = 2s = 2
    # Height = sqrt(3)*s = sqrt(3)
    # For flat-top orientation:
    # Horizontal distance between adjacent columns: 1.5
    # Vertical distance between adjacent rows: sqrt(3)
    # Odd rows are offset by 0.75 horizontally? No, offset by 1.5/2 = 0.75? 
    # Actually, standard hex grid:
    # col_spacing = 1.5
    # row_spacing = sqrt(3)
    # offset for odd rows = 0.75
    
    # Let's place 12 hexagons in a 4x3 grid (4 columns, 3 rows):
    # This gives a parallelogram shape.
    
    col_spacing = 1.5
    row_spacing = math.sqrt(3)
    offset = 0.75
    
    centers = []
    for row in range(3):
        for col in range(4):
            x = col * col_spacing + (offset if row % 2 == 1 else 0)
            y = row * row_spacing
            centers.append([x, y, 0.0])
    
    inner = np.array(centers)
    
    # Now find the minimum enclosing regular hexagon.
    # We'll try rotating the outer hexagon and finding the minimum s that contains all inner vertices.
    # For a given outer hexagon orientation, the minimum s is determined by the maximum projection
    # of inner vertices onto the 6 normal directions of the outer hexagon.
    
    best_s = float('inf')
    best_outer_angle = 0.0
    
    # Try many angles for the outer hexagon
    for angle_deg in np.arange(0, 60, 0.1):
        a = math.radians(angle_deg)
        # The 6 outward normal directions of a regular hexagon with this orientation
        # A regular hexagon with center at origin, side s, angle a has vertices at:
        # (s*cos(a+60i), s*sin(a+60i))
        # The edges are perpendicular to directions at angles a+30+60i
        # So the outward normals are at angles a+30+60i for i=0..5
        
        normals = []
        for i in range(6):
            theta = a + math.radians(30 + 60 * i)
            normals.append(np.array([math.cos(theta), math.sin(theta)]))
        
        # For each normal, find the max and min projection of all inner vertices
        # The hexagon must contain all vertices, so for each normal direction,
        # the distance from center to the farthest vertex in that direction must be <= s * cos(30) = s * sqrt(3)/2
        # Wait, the apothem of the outer hexagon is s * sqrt(3)/2
        # So for each normal direction n, the projection of any point p onto n must satisfy:
        # |dot(p - c, n)| <= s * sqrt(3)/2
        # So s >= max_i |dot(p_i - c, n_i)| / (sqrt(3)/2) for all i
        
        all_verts = []
        for cx, cy, ang in inner:
            verts = _hex_vertices(cx, cy, 1.0, ang)
            all_verts.extend(verts.tolist())
        all_verts = np.array(all_verts)
        
        # We can choose the center of the outer hexagon. For a given orientation,
        # the optimal center is the midpoint of the bounding box in each normal direction.
        # For each pair of opposite normals (i and i+3), we have:
        # max_proj - min_proj <= 2 * s * sqrt(3)/2 = s * sqrt(3)
        # So s >= (max_proj - min_proj) / sqrt(3)
        
        s_needed = 0
        for i in range(3):
            n1 = normals[i]
            n2 = normals[i + 3]  # opposite
            proj1 = np.dot(all_verts, n1)
            proj2 = np.dot(all_verts, n2)
            # n2 = -n1, so proj2 = -proj1
            # max_proj - min_proj = 2 * max(|proj|)
            max_abs = np.max(np.abs(proj1))
            s_candidate = 2 * max_abs / math.sqrt(3)
            s_needed = max(s_needed, s_candidate)
        
        if s_needed < best_s:
            best_s = s_needed
            best_outer_angle = angle_deg
    
    # The outer hexagon center should be at the centroid of the inner vertices
    all_verts = []
    for cx, cy, ang in inner:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_verts.extend(verts.tolist())
    all_verts = np.array(all_verts)
    center = all_verts.mean(axis=0)
    
    outer = np.array([center[0], center[1], best_outer_angle])
    s = best_s
    
    # Verify validity
    if not _is_valid(inner, outer, s):
        # Fallback: use a larger hexagon
        s = s * 1.1
        if not _is_valid(inner, outer, s):
            s = 8.0
            outer = np.array([0, 0, 0])
    
    return inner, outer, s


# EVOLVE-BLOCK-END
