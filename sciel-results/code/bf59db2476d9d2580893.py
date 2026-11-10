# EVOLVE-BLOCK-START
import numpy as np
import time
from math import cos, sin, pi, sqrt

def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of hexagon with center (cx,cy), side d, angle a_deg (CCW)."""
    a_rad = np.radians(a_deg)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.array([[cx + d * cos(angles[i]), cy + d * sin(angles[i])] for i in range(6)])

def _sat_separated(verts_a, verts_b, tol=1e-6):
    """Check if two convex polygons are separated using SAT."""
    for verts in (verts_a, verts_b):
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            proj_a = verts_a @ normal
            proj_b = verts_b @ normal
            if proj_a.max() + tol < proj_b.min() or proj_b.max() + tol < proj_a.min():
                return True
    return False

def _all_separated(inner_verts, tol=1e-6):
    """Check all pairs of inner hexagons are separated."""
    for i in range(len(inner_verts)):
        for j in range(i + 1, len(inner_verts)):
            if not _sat_separated(inner_verts[i], inner_verts[j], tol):
                return False
    return True

def _all_inside(inner_verts, outer_verts, tol=1e-6):
    """Check all inner vertices are inside the outer hexagon."""
    for verts in inner_verts:
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                e = p2 - p1
                pv = v - p1
                cross = e[0] * pv[1] - e[1] * pv[0]
                if cross < -tol:
                    return False
    return True

def _compute_min_s(inner_hex_data, outer_angle_deg):
    """Compute the minimum side length s such that all inner hexagons fit inside
    a regular hexagon with the given angle and center at origin."""
    # Get all inner vertices
    all_verts = []
    for i in range(inner_hex_data.shape[0]):
        cx, cy, angle = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1.0, angle)
        all_verts.append(verts)
    
    # For a regular hexagon with center at origin and angle a,
    # the constraint is that for each vertex v of inner hexagons,
    # the distance from origin projected onto each of the 6 face normals
    # must be <= s * apothem_factor
    # Actually, for a regular hexagon with side s and angle a,
    # the distance from center to each edge (apothem) is s * sqrt(3)/2
    # The 6 face normals are at angles a + 30 + 60*i degrees
    
    # For a point to be inside, for each face normal direction,
    # the dot product of the point with the normal must be <= apothem
    # apothem = s * sqrt(3) / 2
    
    a_rad = np.radians(outer_angle_deg)
    # Face normals are perpendicular to edges, at angles a + 30 + 60*i
    normal_angles = a_rad + np.pi/6 + np.arange(6) * np.pi / 3
    normals = np.array([[cos(na), sin(na)] for na in normal_angles])
    
    max_proj = 0
    for verts in all_verts:
        for v in verts:
            projs = verts @ normals.T  # Wait, need to do for each v
            # Actually, compute projection of each vertex onto each normal
            for v in verts:
                projs = v @ normals.T
                max_proj = max(max_proj, projs.max())
    
    # apothem = s * sqrt(3)/2, so s = 2 * apothem / sqrt(3)
    s = 2.0 * max_proj / sqrt(3)
    return s

def _is_valid(inner_hex_data, outer_hex_data, s, tol=1e-6):
    """Full validity check."""
    if inner_hex_data.shape != (12, 3) or outer_hex_data.shape != (3,):
        return False
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < sqrt(12) - 1e-9:
        return False
    
    ox, oy, oa = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, s, oa)
    
    # Check distinct outer vertices
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    # Compute inner vertices
    inner_verts = []
    for i in range(12):
        cx, cy, angle = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1.0, angle)
        inner_verts.append(verts)
    
    # Check separation
    if not _all_separated(inner_verts, tol):
        return False
    
    # Check containment
    if not _all_inside(inner_verts, outer_verts, tol):
        return False
    
    return True

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 seconds safety margin
    
    # Initial arrangement: hexagonal grid
    # Center + 6 in ring 1 + 5 in ring 2
    sqrt3 = sqrt(3)
    
    # Ring 0: center
    centers = [(0, 0)]
    
    # Ring 1: 6 hexagons at distance sqrt(3) from center
    for k in range(6):
        a = 2 * pi * k / 6
        centers.append((sqrt3 * cos(a), sqrt3 * sin(a)))
    
    # Ring 2: We need 5 more. Place them at the "corner" positions of ring 2
    # Ring 2 positions are at distance 2*sqrt(3) from center, at angles 0, 60, 120, 180, 240, 300
    # But we only need 5. Let's place them at angles 0, 60, 120, 180, 240 (skip 300 for symmetry breaking)
    # Actually, for a symmetric arrangement, let's try placing 5 at specific positions.
    # A better approach: use the actual hex grid coordinates.
    
    # Hex grid: axial coordinates (q, r)
    # Position: x = sqrt(3) * (q + r/2), y = 1.5 * r
    # Let's enumerate a compact set of 12 positions.
    
    # Use a known compact arrangement for 12 hexagons
    # Positions in axial coordinates (q, r):
    positions_axial = [
        (0, 0),      # 0
        (1, 0),      # 1
        (0, 1),      # 2
        (-1, 1),     # 3
        (-1, 0),     # 4
        (0, -1),     # 5
        (1, -1),     # 6
        (2, 0),      # 7
        (1, 1),      # 8
        (-2, 1),     # 9
        (-2, 0),     # 10
        (-1, -1),    # 11
    ]
    
    centers = []
    for q, r in positions_axial:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    inner_hex_data = np.array([(cx, cy, 0) for cx, cy in centers])
    
    # Compute initial s
    best_s = _compute_min_s(inner_hex_data, 0)
    best_inner = inner_hex_data.copy()
    best_outer = np.array([0, 0, 0])
    
    # Local search: perturb positions and adjust outer angle
    np.random.seed(42)
    current_inner = best_inner.copy()
    current_s = best_s
    current_outer_angle = 0.0
    
    # Try different outer angles
    for outer_angle in [0, 30, 60, 90, 120, 150]:
        s_test = _compute_min_s(current_inner, outer_angle)
        if s_test < best_s:
            best_s = s_test
            best_outer_angle = outer_angle
    
    best_outer_angle = 0.0
    for outer_angle in [0, 30, 60, 90, 120, 150]:
        s_test = _compute_min_s(current_inner, outer_angle)
        if s_test < best_s:
            best_s = s_test
            best_outer_angle = outer_angle
    
    # Iterative local search
    step_size = 0.1
    for iteration in range(20000):
        if time.time() > deadline - 30:
            break
        
        # Randomly perturb one hexagon's position
        idx = np.random.randint(0, 12)
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        
        old_x, old_y, old_a = current_inner[idx]
        new_x = old_x + dx
        new_y = old_y + dy
        current_inner[idx] = [new_x, new_y, old_a]
        
        # Compute new s
        new_s = _compute_min_s(current_inner, best_outer_angle)
        
        # Check validity
        outer_verts = _hex_vertices(0, 0, new_s, best_outer_angle)
        inner_verts = [_hex_vertices(current_inner[i][0], current_inner[i][1], 1.0, current_inner[i][2]) 
                      for i in range(12)]
        
        valid = _all_separated(inner_verts, 1e-6) and _all_inside(inner_verts, outer_verts, 1e-6)
        
        if valid and new_s < current_s:
            current_s = new_s
            if new_s < best_s:
                best_s = new_s
                best_inner = current_inner.copy()
        else:
            # Revert
            current_inner[idx] = [old_x, old_y, old_a]
        
        # Occasionally adjust outer angle
        if iteration % 100 == 0:
            for angle in [best_outer_angle - 5, best_outer_angle, best_outer_angle + 5]:
                s_test = _compute_min_s(current_inner, angle)
                if s_test < best_s:
                    best_s = s_test
                    best_outer_angle = angle
    
    # Final result
    best_outer = np.array([0, 0, best_outer_angle])
    
    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END
