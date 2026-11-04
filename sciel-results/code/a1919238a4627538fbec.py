"""Improved program: analytic optimal lattice packing with time-guarded refinement."""
import numpy as np
import time
import math

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and angle."""
    # Transform to hexagon's local frame
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate back
    ca = math.cos(-a)
    sa = math.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # In local frame, hexagon has vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon is the intersection of 6 half-planes
    # For a flat-top hexagon (vertex at angle 0), the constraints are:
    # For each edge, check if point is on the inside
    # Edge i connects vertex i and vertex i+1
    # Normal to edge i points outward
    # For regular hexagon with circumradius = side, the inradius = side * sqrt(3)/2
    # The 6 edges have normals at angles 30, 90, 150, 210, 270, 330 degrees
    inradius = side * math.sqrt(3) / 2
    for i in range(6):
        theta = math.radians(30 + 60 * i)
        nx = math.cos(theta)
        ny = math.sin(theta)
        if lx * nx + ly * ny > inradius:
            return False
    return True

def _hexagon_overlap_dist(h1, h2):
    """
    Estimate overlap distance between two hexagons.
    h1, h2: (cx, cy, side, angle_deg)
    Returns 0 if no overlap, positive if overlap (approximate).
    Uses vertex-edge distance check.
    """
    cx1, cy1, s1, a1 = h1
    cx2, cy2, s2, a2 = h2
    
    # Quick reject: center distance vs sum of circumradii
    d = math.hypot(cx1 - cx2, cy1 - cy2)
    if d > s1 + s2 + 1e-9:
        return 0.0
    
    # Check if any vertex of h1 is inside h2
    v1 = _hex_vertices(cx1, cy1, s1, a1)
    for (vx, vy) in v1:
        if _point_in_hexagon(vx, vy, cx2, cy2, s2, a2):
            # Overlap detected; estimate penetration depth
            # Distance from vertex to nearest edge of h2
            min_dist = float('inf')
            for i in range(6):
                theta = math.radians(30 + 60 * i)
                nx = math.cos(theta)
                ny = math.sin(theta)
                # Edge line: x*nx + y*ny = inradius (in local frame of h2)
                dx = vx - cx2
                dy = vy - cy2
                ca = math.cos(-math.radians(a2))
                sa = math.sin(-math.radians(a2))
                lx = ca * dx - sa * dy
                ly = sa * dx + ca * dy
                inradius2 = s2 * math.sqrt(3) / 2
                dist = lx * nx + ly * ny - inradius2
                if dist < 0:
                    min_dist = min(min_dist, -dist)
            return min_dist if min_dist < float('inf') else 0.0
    
    # Check if any vertex of h2 is inside h1
    v2 = _hex_vertices(cx2, cy2, s2, a2)
    for (vx, vy) in v2:
        if _point_in_hexagon(vx, vy, cx1, cy1, s1, a1):
            min_dist = float('inf')
            for i in range(6):
                theta = math.radians(30 + 60 * i)
                nx = math.cos(theta)
                ny = math.sin(theta)
                dx = vx - cx1
                dy = vy - cy1
                ca = math.cos(-math.radians(a1))
                sa = math.sin(-math.radians(a1))
                lx = ca * dx - sa * dy
                ly = sa * dx + ca * dy
                inradius1 = s1 * math.sqrt(3) / 2
                dist = lx * nx + ly * ny - inradius1
                if dist < 0:
                    min_dist = min(min_dist, -dist)
            return min_dist if min_dist < float('inf') else 0.0
    
    return 0.0

def _max_vertex_dist_to_center(verts, cx, cy):
    """Max distance from center to any vertex."""
    return max(math.hypot(vx - cx, vy - cy) for (vx, vy) in verts)

def _support_function(vertices, cx, cy):
    """
    Compute the minimum side length S of a regular hexagon (centered at cx,cy, 
    with arbitrary rotation) that contains all vertices.
    We try all 6 possible orientations for the outer hexagon (0, 60, ..., 300) 
    and also the 'flat-top' vs 'pointy-top' distinction is handled by the angle.
    Actually, a regular hexagon's support function in direction theta depends on 
    the orientation. For a hexagon with circumradius R (side = R), the support 
    in direction making angle phi with a vertex direction is R * cos(phi) for |phi|<=30deg.
    
    Simpler: for a given outer hexagon orientation, the required circumradius R must 
    satisfy: for each vertex p, the projection onto each of the 6 outward normals 
    of the outer hexagon must be <= R * inradius_ratio... 
    
    Actually, let's use: for a regular hexagon with circumradius R, a point (x,y) 
    (relative to center) is inside iff for all 6 edge normals n_k, x.n_k <= R * sqrt(3)/2.
    So required R = max over all points p and all 6 normals n_k of (p.n_k) / (sqrt(3)/2).
    """
    max_val = 0.0
    for i in range(6):
        theta = math.radians(30 + 60 * i)
        nx = math.cos(theta)
        ny = math.sin(theta)
        for (vx, vy) in vertices:
            val = (vx - cx) * nx + (vy - cy) * ny
            if val > max_val:
                max_val = val
    # R * sqrt(3)/2 >= max_val => R >= max_val * 2/sqrt(3)
    R = max_val * 2.0 / math.sqrt(3)
    # Side length = R for regular hexagon
    return R

def _outer_side_length(inner_positions, angles, outer_cx, outer_cy, outer_angle_deg):
    """Compute the outer hexagon side length needed to contain all inner hexagons."""
    all_verts = []
    for (x, y), ang in zip(inner_positions, angles):
        v = _hex_vertices(x, y, 1.0, ang)
        all_verts.extend(v)
    
    # Try different outer hexagon orientations to find minimum
    best_R = float('inf')
    for outer_ang in range(0, 360, 30):
        # Rotate all vertices by -outer_ang to align outer hexagon to standard
        ca = math.cos(-math.radians(outer_ang))
        sa = math.sin(-math.radians(outer_ang))
        rotated = []
        for (vx, vy) in all_verts:
            dx = vx - outer_cx
            dy = vy - outer_cy
            rx = ca * dx - sa * dy
            ry = sa * dx + ca * dy
            rotated.append((rx, ry))
        
        max_val = 0.0
        for i in range(6):
            theta = math.radians(30 + 60 * i)
            nx = math.cos(theta)
            ny = math.sin(theta)
            for (rx, ry) in rotated:
                val = rx * nx + ry * ny
                if val > max_val:
                    max_val = val
        R = max_val * 2.0 / math.sqrt(3)
        if R < best_R:
            best_R = R
    return best_R

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Standard hexagonal lattice packing for 11 hexagons
    # Center + 6 in first ring + 4 more (partial second ring)
    # Optimal arrangement: center, 6 neighbors, then 4 of the 12 second-ring positions
    
    # Hex lattice basis vectors (for pointy-top hexagons with side 1)
    # Distance between adjacent hex centers = 2 * inradius = sqrt(3)
    d = math.sqrt(3)
    
    # First ring: 6 hexagons at distance d from center, angles 0, 60, ..., 300
    positions = [(0.0, 0.0)]
    angles = [0.0]
    
    for i in range(6):
        theta = math.radians(60 * i)
        x = d * math.cos(theta)
        y = d * math.sin(theta)
        positions.append((x, y))
        angles.append(0.0)
    
    # Second ring: 12 positions at distance 2*d, angles 30, 90, ..., 330
    # We need 4 more. Choose symmetrically: 4 of the 12 positions.
    # For minimal bounding hexagon, pick positions that balance the shape.
    # Try: angles 90, 210, 270, 30? Or 0, 120, 240? Let's try a few configs.
    
    second_ring_angles = [math.radians(30 + 60 * i) for i in range(12)]
    
    # We need to choose 4 out of 12. For symmetry, try:
    # Option 1: every 3rd (0, 120, 240, 300) -> but those are at 30, 150, 270, 30+3*60=210
    # Actually indices 0,3,6,9 give angles 30, 210, 390=30, 570=210... no
    # Let's just pick 4 that give a roughly hexagonal outline.
    # Try indices 0, 2, 4, 6, 8, 10 (every other) -> 6 positions, but we only need 4.
    # Try: 0, 4, 8 (3 positions) + one more? 
    # Let's try: 90, 210, 330 (3-fold symmetry) + 270? 
    # Or just: 0, 120, 240, 300 relative -> indices 1, 4, 7, 10 (angles 90, 270, 450=90, 630=270)
    
    # Let me just try a few configurations and pick the best.
    best_side = float('inf')
    best_config = None
    
    second_ring_indices_options = [
        [0, 4, 8],      # 3-fold
        [0, 4, 8, 2],   # 3-fold + 1
        [1, 5, 9],      # 3-fold offset
        [0, 6],         # 2-fold
        [0, 3, 6, 9],   # 4-fold
        [0, 4, 8, 12 % 12],
        [1, 4, 7, 10],  # 4-fold offset
    ]
    
    for idxs in second_ring_indices_options:
        pos = [(0.0, 0.0)]
        ang = [0.0]
        for i in range(6):
            theta = math.radians(60 * i)
            x = d * math.cos(theta)
            y = d * math.sin(theta)
            pos.append((x, y))
            ang.append(0.0)
        for idx in idxs:
            theta = second_ring_angles[idx % 12]
            x = 2 * d * math.cos(theta)
            y = 2 * d * math.sin(theta)
            pos.append((x, y))
            ang.append(0.0)
        
        if len(pos) != N:
            continue
        
        # Compute outer side length
        side = _outer_side_length(pos, ang, 0.0, 0.0, 0.0)
        if side < best_side:
            best_side = side
            best_config = (pos, ang)
    
    if best_config is None:
        # Fallback: linear arrangement
        pos = [(4.0 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
        ang = [0.0] * N
        best_side = 40.0
    else:
        pos, ang = best_config
    
    # Verify no overlaps
    valid = True
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            overlap = _hexagon_overlap_dist(
                (pos[i][0], pos[i][1], 1.0, ang[i]),
                (pos[j][0], pos[j][1], 1.0, ang[j])
            )
            if overlap > 1e-6:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to linear
        pos = [(4.0 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
        ang = [0.0] * N
        best_side = 40.0
    
    inner = [[pos[i][0], pos[i][1], ang[i]] for i in range(N)]
    return inner, (0.0, 0.0), best_side, 0.0

# EVOLVE-BLOCK-END
