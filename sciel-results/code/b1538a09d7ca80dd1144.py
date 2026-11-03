"""Rewritten program: pack N unit hexagons in a minimal regular hexagon."""
import time
import math
import numpy as np

N = 11

# Time budget management
_START_TIME = time.time()
TIME_LIMIT = 120.0  # seconds, with safety margin we use 115

def _elapsed():
    return time.time() - _START_TIME

def _hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform to local coords
    dx = px - cx
    dy = py - cy
    angle_rad = math.radians(angle_deg)
    ca, sa = math.cos(-angle_rad), math.sin(-angle_rad)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # In local coords, hexagon has vertices at angles 0, 60, ..., 300
    # A regular hexagon with side s has inradius r = s * sqrt(3)/2
    # The hexagon in local coords: |lx| <= s, and for the y-direction:
    # Actually, for a regular hexagon with "flat top" orientation (vertex at angle 0):
    # The boundaries are: |ly| <= s * sqrt(3)/2
    # and |lx| + |ly| / sqrt(3) <= s  ... let me think more carefully.
    
    # For a regular hexagon with vertices at (s*cos(k*pi/3), s*sin(k*pi/3)) for k=0..5:
    # The six edges are at distances s*sqrt(3)/2 from center.
    # The support function in direction theta is h(theta) = s * sqrt(3)/2 * max over ... 
    # Actually simpler: a point (lx, ly) is inside if:
    # |ly| <= s * sqrt(3)/2
    # |lx| <= s - |ly| / sqrt(3)  (when |ly| <= s*sqrt(3)/2)
    # Wait, that's for a hexagon with flat sides horizontal. Let me verify.
    
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees:
    # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    
    # The top edge connects (s/2, s*sqrt(3)/2) and (-s/2, s*sqrt(3)/2), so y = s*sqrt(3)/2
    # The upper-right edge connects (s, 0) and (s/2, s*sqrt(3)/2)
    # Line through these two points: slope = (s*sqrt(3)/2 - 0)/(s/2 - s) = (s*sqrt(3)/2)/(-s/2) = -sqrt(3)
    # Equation: y - 0 = -sqrt(3)(x - s) => y = -sqrt(3)x + s*sqrt(3) => sqrt(3)x + y = s*sqrt(3)
    # So the constraint is: sqrt(3)*lx + ly <= s*sqrt(3)
    # Similarly for other edges:
    # sqrt(3)*lx - ly <= s*sqrt(3)
    # -sqrt(3)*lx + ly <= s*sqrt(3)
    # -sqrt(3)*lx - ly <= s*sqrt(3)
    # lx <= s  (from right vertex, but this is implied by the above when ly=0)
    # Actually let's just use the 6 half-plane constraints:
    # For each edge normal direction, the dot product with the point must be <= s*sqrt(3)/2
    
    # The 6 outward normals are at angles 30, 90, 150, 210, 270, 330 degrees (perpendicular to edges)
    # Edge between vertices at 0 and 60 deg: midpoint direction is 30 deg, normal is 30 deg
    # Distance from center to each edge = s * sqrt(3)/2
    
    r = s * math.sqrt(3) / 2  # inradius
    for i in range(6):
        theta = math.pi / 6 + i * math.pi / 3  # 30, 90, 150, 210, 270, 330
        nx, ny = math.cos(theta), math.sin(theta)
        if lx * nx + ly * ny > r + 1e-9:
            return False
    return True

def _hexagon_overlap(v1, v2, v3, v4, tol=1e-9):
    """Check if two convex hexagons overlap using separating axis theorem."""
    # Get all edges from both polygons
    edges = []
    for poly in (v1, v2):
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i + 1) % 6]
            # Edge vector
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            # Normal
            nx, ny = -ey, ex
            norm = math.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                edges.append((nx/norm, ny/norm))
    
    for nx, ny in edges:
        # Project all points of both polygons onto this axis
        min1, max1 = float('inf'), float('-inf')
        min2, max2 = float('inf'), float('-inf')
        for p in v1:
            proj = p[0]*nx + p[1]*ny
            min1 = min(min1, proj)
            max1 = max(max1, proj)
        for p in v2:
            proj = p[0]*nx + p[1]*ny
            min2 = min(min2, proj)
            max2 = max(max2, proj)
        # Check for separation
        if max1 <= min2 + tol or max2 <= min1 + tol:
            return False
    return True

def _hexagon_inside_hexagon(inner_verts, outer_verts):
    """Check if inner hexagon is inside outer hexagon using SAT."""
    # Check all edges of both polygons
    edges = []
    for poly in (inner_verts, outer_verts):
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i + 1) % 6]
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            nx, ny = -ey, ex
            norm = math.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                edges.append((nx/norm, ny/norm))
    
    for nx, ny in edges:
        min_in, max_in = float('inf'), float('-inf')
        min_out, max_out = float('inf'), float('-inf')
        for p in inner_verts:
            proj = p[0]*nx + p[1]*ny
            min_in = min(min_in, proj)
            max_in = max(max_in, proj)
        for p in outer_verts:
            proj = p[0]*nx + p[1]*ny
            min_out = min(min_out, proj)
            max_out = max(max_out, proj)
        # Inner must be within outer on this axis
        if min_in < min_out - 1e-9 or max_in > max_out + 1e-9:
            return False
    return True

def _compute_outer_side(positions, angles):
    """Compute minimum side length of a regular hexagon (with some rotation) that contains all inner hexagons.
    Returns (side_length, outer_center, outer_rotation)."""
    # For a given set of inner hexagons, we need to find the minimum regular hexagon containing them.
    # The outer hexagon has 6 degrees of freedom: cx, cy, side, angle.
    # But we can fix cx, cy to be the centroid of all inner hexagon centers for optimality.
    # Actually, the optimal center is not necessarily the centroid. Let's think differently.
    
    # A regular hexagon with side S and rotation alpha has support function:
    # h(theta) = S * sqrt(3)/2 * max over 6 directions... 
    # Actually, the support function of a regular hexagon with inradius r = S*sqrt(3)/2 is:
    # h(theta) = r / cos((theta - alpha - 30*pi/6) mod 60*pi/6 - 30*pi/6) ... this is getting complex.
    
    # Simpler approach: for a given outer rotation alpha, the constraint for each inner hexagon vertex (px, py)
    # is that it must be inside the outer hexagon. The outer hexagon with center (cx,cy), side S, rotation alpha
    # contains point (px,py) iff for all 6 edge normals:
    # (px-cx)*nx_i + (py-cy)*ny_i <= S*sqrt(3)/2
    # where nx_i, ny_i are the 6 outward normals of the outer hexagon.
    
    # So S >= max over all inner vertices and all 6 normals of: ((px-cx)*nx_i + (py-cy)*ny_i) / (sqrt(3)/2)
    
    # To minimize S, we need to choose cx, cy, alpha to minimize this max.
    # For a fixed alpha, the optimal cx, cy can be found by noting that:
    # S = max_i [max_v ((vx - cx)*nx_i + (vy - cy)*ny_i)] / (sqrt(3)/2)
    # = max_i [max_v (vx*nx_i + vy*ny_i) - cx*nx_i - cy*ny_i] / (sqrt(3)/2)
    
    # This is a convex optimization in (cx, cy) for fixed alpha. We can use a simple approach:
    # The optimal (cx, cy) minimizes max over 6 directions of (max_v dot(v, n_i) - dot(c, n_i))
    # This is equivalent to finding the minimum enclosing regular hexagon.
    
    # For simplicity and speed, let's try: fix alpha to a few values, and for each, 
    # use the centroid as a starting point and do a few gradient-free iterations.
    
    # Actually, let's use a simpler but effective approach:
    # 1. Compute all vertices of all inner hexagons
    # 2. For each candidate outer rotation alpha (we'll try a few), find the minimum S
    #    by binary search or direct computation.
    
    # For a fixed alpha and center (cx, cy):
    # S_min = max over all vertices v and all 6 normals n_i of (v·n_i - c·n_i) / (sqrt(3)/2)
    
    # The optimal center for fixed alpha: we want to minimize max_i (M_i - c·n_i)
    # where M_i = max_v (v·n_i)
    # This is a Chebyshev-type problem. The optimal c satisfies:
    # c·n_i = (M_i + M_{i+3})/2 for opposite normals (since n_{i+3} = -n_i)
    # Wait, for a regular hexagon, the normals come in 3 opposite pairs.
    # n_0 and n_3 are opposite, n_1 and n_4 are opposite, n_2 and n_5 are opposite.
    # So we have 3 independent directions.
    # Let's define for each pair k=0,1,2:
    #   a_k = M_{2k} (max projection on n_{2k})
    #   b_k = M_{2k+3} = -min projection on n_{2k} = -m_{2k} where m is min
    #   Actually M_{2k+3} = max_v (v·(-n_{2k})) = -min_v (v·n_{2k})
    # So the constraint in direction k is: c·n_k >= M_{2k} - S*sqrt(3)/2 and c·n_k <= -M_{2k+3} + S*sqrt(3)/2
    # Wait, I'm overcomplicating this.
    
    # Let me just do: for each alpha, compute all 12 projections (6 normals, max and min over all vertices)
    # Then S = max over 6 normals of (max_proj - min_proj) / (2 * sqrt(3)/2) ... no.
    
    # For a fixed alpha, the minimum S hexagon centered at (cx, cy) requires:
    # For each of the 6 normals n_i: cx*nx_i + cy*ny_i >= max_v(v·n_i) - S*sqrt(3)/2
    # And: cx*nx_i + cy*ny_i <= min_v(v·n_i) + S*sqrt(3)/2  (wait, this isn't right either)
    
    # Let me restart this calculation.
    # Outer hexagon: center c, side S, rotation alpha.
    # The 6 outward unit normals are u_i = (cos(alpha + 30° + i*60°), sin(alpha + 30° + i*60°)) for i=0..5
    # A point p is inside iff (p - c) · u_i <= S*sqrt(3)/2 for all i.
    # So for all inner vertices v: (v - c) · u_i <= S*sqrt(3)/2 for all i.
    # => c · u_i >= v · u_i - S*sqrt(3)/2 for all v, i
    # => c · u_i >= max_v(v · u_i) - S*sqrt(3)/2 for all i
    # Let M_i = max_v(v · u_i). Then: c · u_i >= M_i - S*sqrt(3)/2
    # Also, for the opposite normal u_{i+3} = -u_i:
    # c · (-u_i) >= M_{i+3} - S*sqrt(3)/2
    # => -c · u_i >= M_{i+3} - S*sqrt(3)/2
    # => c · u_i <= -M_{i+3} + S*sqrt(3)/2
    # Note M_{i+3} = max_v(v · (-u_i)) = -min_v(v · u_i) = -m_i where m_i = min_v(v·u_i)
    # So: c · u_i <= m_i + S*sqrt(3)/2
    # Combining: M_i - S*sqrt(3)/2 <= c·u_i <= m_i + S*sqrt(3)/2
    # This requires: M_i - S*sqrt(3)/2 <= m_i + S*sqrt(3)/2
    # => M_i - m_i <= S*sqrt(3)
    # => S >= (M_i - m_i) / sqrt(3)
    
    # So for fixed alpha: S_min = max_i (M_i - m_i) / sqrt(3)
    # And the center c must satisfy: M_i - S*sqrt(3)/2 <= c·u_i <= m_i + S*sqrt(3)/2 for all i.
    # The feasible region for c is a hexagon (intersection of 6 strips).
    # We can pick c as the centroid of the inner hexagon centers as a reasonable choice.
    
    # Let's implement this:
    all_verts = []
    for (px, py), ang in zip(positions, angles):
        verts = _hex_vertices(px, py, 1.0, ang)
        all_verts.extend(verts)
    all_verts = np.array(all_verts)
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_alpha = 0.0
    
    # Try several outer rotations
    for alpha_deg in range(0, 60, 5):  # 0, 5, 10, ..., 55
        alpha = math.radians(alpha_deg)
        # Compute 6 normals
        normals = []
        for i in range(6):
            theta = alpha + math.pi/6 + i * math.pi/3
            normals.append((math.cos(theta), math.sin(theta)))
        
        # Compute M_i and m_i for each normal
        M = []
        m = []
        for nx, ny in normals:
            projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            M.append(np.max(projs))
            m.append(np.min(projs))
        
        S_candidate = max((M[i] - m[i]) for i in range(6)) / math.sqrt(3)
        
        if S_candidate < best_side:
            best_side = S_candidate
            best_alpha = alpha_deg
            # Find a valid center: use the midpoint of the feasible region
            # c·u_i should be between M_i - S*sqrt(3)/2 and m_i + S*sqrt(3)/2
            # A reasonable choice: c·u_i = (M_i + m_i) / 2 ... but this needs to be consistent.
            # Actually, let's just use the centroid of all vertices as an approximation.
            cx = np.mean(all_verts[:, 0])
            cy = np.mean(all_verts[:, 1])
            best_center = (cx, cy)
    
    return best_side, best_center, best_alpha

def _check_validity(positions, angles, outer_center, outer_side, outer_angle):
    """Check if the packing is valid: no overlaps, all inside outer hexagon."""
    # Get all inner hexagon vertices
    inner_verts_list = []
    for (px, py), ang in zip(positions, angles):
        verts = _hex_vertices(px, py, 1.0, ang)
        inner_verts_list.append(verts)
    
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Check containment
    for verts in inner_verts_list:
        if not _hexagon_inside_hexagon(verts, outer_verts):
            return False
    
    # Check non-overlap
    for i in range(len(inner_verts_list)):
        for j in range(i+1, len(inner_verts_list)):
            if _hexagon_overlap(inner_verts_list[i], inner_verts_list[j]):
                return False
    
    return True

# EVOLVE-BLOCK-START
def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Strategy: use a hexagonal lattice arrangement for the inner hexagons,
    # then optimize the global rotation and translation.
    
    # For N=11, a good arrangement is a hexagonal cluster:
    # Center + 6 around + 4 more (or similar)
    # Standard hexagonal packing: 1 + 6 + 12 = 19 for a full hexagonal number
    # For 11: 1 + 6 + 4 (partial second ring)
    
    # Let's use a hexagonal lattice with spacing 2*sqrt(3) for side-1 hexagons
    # (adjacent hexagons in a honeycomb have center distance 2*sqrt(3)/... 
    # Actually, for regular hexagons of side 1, the distance between centers of adjacent
    # hexagons in a honeycomb is 2 * inradius = 2 * sqrt(3)/2 = sqrt(3) * ... 
    # Wait: inradius = sqrt(3)/2, so center-to-center for touching hexagons = 2 * inradius = sqrt(3)
    # No: two regular hexagons of side s touching edge-to-edge have center distance 2 * inradius = 2 * s*sqrt(3)/2 = s*sqrt(3)
    # For s=1: distance = sqrt(3) ≈ 1.732
    
    # Hexagonal lattice basis:
    # e1 = (sqrt(3), 0)  -- horizontal neighbor
    # e2 = (sqrt(3)/2, 3/2) -- 60 degree neighbor
    # Actually, the three nearest neighbors are at 120 degrees apart.
    # For a honeycomb, the lattice vectors are:
    # a1 = (3/2, sqrt(3)/2) * 2 = (3, sqrt(3)) ... let me be more careful.
    
    # Two unit hexagons sharing an edge: the centers are separated by 2*inradius = sqrt(3)
    # in the direction perpendicular to the shared edge.
    # The 6 neighbors of a hexagon are at angles 0, 60, 120, 180, 240, 300 (depending on orientation)
    # at distance sqrt(3) from the center.
    
    # So the hexagonal lattice has basis:
    # a1 = sqrt(3) * (1, 0) = (sqrt(3), 0)
    # a2 = sqrt(3) * (cos(60°), sin(60°)) = sqrt(3) * (1/2, sqrt(3)/2) = (sqrt(3)/2, 3/2)
    
    sqrt3 = math.sqrt(3)
    a1 = (sqrt3, 0.0)
    a2 = (sqrt3/2, 1.5)
    
    # Generate hexagonal lattice points for a cluster of 11
    # Ring 0: (0,0) -> 1 point
    # Ring 1: 6 points at distance sqrt(3)
    # Ring 2: 12 points, take 4 of them
    # Total: 1 + 6 + 4 = 11
    
    # Actually, let's use a more compact arrangement.
    # For 11 hexagons, we can use:
    # Row-based arrangement in a hexagonal grid.
    
    # Let me define positions in the hexagonal lattice coordinates (i, j) where
    # position = i * a1 + j * a2
    
    # A compact cluster of 11:
    # Center at (0,0)
    # 6 neighbors: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # 4 more from the next ring: (2,0), (-2,0), (0,2), (0,-2) -- but these might be too far
    # Let's try: (2,-1), (2,0), (1,1), (0,2) or similar
    
    # Let me just enumerate a good set of 11 lattice points:
    lattice_points = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, -1), (2, 0), (1, 1), (0, 2)
    ]
    # That's 11 points. Let me verify they form a valid cluster.
    
    # Convert to Cartesian
    base_positions = []
    for i, j in lattice_points:
        x = i * a1[0] + j * a2[0]
        y = i * a1[1] + j * a2[1]
        base_positions.append((x, y))
    
    # All inner hexagons have the same orientation (0 degrees) initially
    base_angles = [0.0] * N
    
    # Compute initial outer side
    side, center, angle = _compute_outer_side(base_positions, base_angles)
    
    # Try to improve by rotating the entire configuration
    best_result = (base_positions, base_angles, center, side, angle)
    
    # Try global rotations
    for rot_deg in range(0, 60, 3):
        rot_rad = math.radians(rot_deg)
        cr, sr = math.cos(rot_rad), math.sin(rot_rad)
        rotated_positions = []
        for (x, y) in base_positions:
            rx = cr * x - sr * y
            ry = sr * x + cr * y
            rotated_positions.append((rx, ry))
        
        rotated_angles = [(a + rot_deg) % 360 for a in base_angles]
        
        s, c, a = _compute_outer_side(rotated_positions, rotated_angles)
        if s < best_result[3] - 1e-9:
            best_result = (rotated_positions, rotated_angles, c, s, a)
    
    # Also try small translations
    best_positions, best_angles, best_center, best_side, best_angle = best_result
    
    # Try translating the whole configuration
    for dx in [-0.5, -0.3, 0.0, 0.3, 0.5]:
        for dy in [-0.5, -0.3, 0.0, 0.3, 0.5]:
            translated_positions = [(x + dx, y + dy) for (x, y) in best_positions]
            s, c, a = _compute_outer_side(translated_positions, best_angles)
            if s < best_side - 1e-9:
                best_positions = translated_positions
                best_center = c
                best_side = s
                best_angle = a
    
    # Verify and if invalid, fall back to a safe large hexagon
    if _check_validity(best_positions, best_angles, best_center, best_side, best_angle):
        inner = [[x, y, ang] for (x, y), ang in zip(best_positions, best_angles)]
        return inner, best_center, best_side, best_angle
    else:
        # Fallback: spread them out more
        fallback_positions = []
        for i, j in lattice_points:
            x = i * a1[0] * 1.05 + j * a2[0] * 1.05
            y = i * a1[1] * 1.05 + j * a2[1] * 1.05
            fallback_positions.append((x, y))
        s, c, a = _compute_outer_side(fallback_positions, base_angles)
        inner = [[x, y, 0.0] for (x, y) in fallback_positions]
        return inner, c, s, a
# EVOLVE-BLOCK-END
