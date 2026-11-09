"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Hexagonal lattice packing for N unit hexagons
    # Unit hexagon: circumradius = 1, inradius = sqrt(3)/2
    # Minimum center-to-center distance for non-overlapping (aligned) = 2.0
    
    # Generate hexagonal grid positions using axial coordinates
    # Convert axial (q, r) to Cartesian: x = 2*q + r, y = sqrt(3)*r  (for pointy-top)
    # Or for flat-top: x = 1.5*q, y = sqrt(3)*(r + q/2)
    # Let's use: for hex grid with spacing 2, positions:
    # x = 2*q + r, y = sqrt(3)*r gives hex centers for unit hexagons
    
    sqrt3 = math.sqrt(3)
    
    # Generate candidate positions in hexagonal grid
    # Axial coordinates (q, r) -> Cartesian (x, y) = (2*q + r, sqrt3 * r)
    # This gives a hex grid where adjacent hexagons are distance 2 apart
    
    def axial_to_cartesian(q, r):
        x = 2.0 * q + r
        y = sqrt3 * r
        return x, y
    
    # Generate positions in a hexagonal cluster around origin
    # Ring 0: (0,0)
    # Ring 1: 6 positions
    # Ring 2: 12 positions
    # etc.
    
    def hex_ring_positions(ring):
        if ring == 0:
            return [(0, 0)]
        positions = []
        # Start at (ring, 0) and go around
        q, r = ring, 0
        for side in range(6):
            for step in range(ring):
                positions.append((q, r))
                # Move in direction 'side'
                dq = [1, 0, -1, -1, 0, 1][side]
                dr = [0, -1, -1, 0, 1, 1][side]
                q += dq
                r += dr
        return positions
    
    # Collect positions by ring
    all_positions = [(0, 0)]
    ring = 1
    while len(all_positions) < N:
        all_positions.extend(hex_ring_positions(ring))
        ring += 1
    
    # Take first N positions
    selected = all_positions[:N]
    
    # Convert to Cartesian
    inner = []
    for q, r in selected:
        x, y = axial_to_cartesian(q, r)
        inner.append([x, y, 0.0])  # angle 0
    
    # Compute minimum enclosing hexagon side length
    # For a hexagon centered at (cx, cy) with side S and rotation angle,
    # check if all vertices of all inner hexagons are inside.
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of regular hexagon with given center, side, rotation."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts
    
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside hexagon centered at (cx,cy) with side, angle."""
        # Translate to center
        dx = px - cx
        dy = py - cy
        # Rotate by -angle to align hexagon with standard orientation
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        
        # Standard hexagon (vertex at angle 0) with circumradius = side
        # Check 6 half-plane constraints
        # Normals at angles 30, 90, 150, 210, 270, 330 degrees
        apothem = side * sqrt3 / 2
        for k in range(6):
            theta = math.radians(30 + k * 60)
            nx = math.cos(theta)
            ny = math.sin(theta)
            if rx * nx + ry * ny > apothem + 1e-9:
                return False
        return True
    
    def inner_hex_in_outer(inner_center, inner_angle, outer_center, outer_side, outer_angle):
        """Check if unit hexagon at inner_center with inner_angle is inside outer hexagon."""
        ix, iy = inner_center
        # Get vertices of inner hexagon
        inner_verts = hexagon_vertices(ix, iy, 1.0, inner_angle)
        for vx, vy in inner_verts:
            if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                return False
        return True
    
    def check_valid(inner_list, outer_center, outer_side, outer_angle):
        """Check all inner hexagons are inside outer and don't overlap."""
        cx, cy = outer_center
        # Check each inner is inside outer
        for ix, iy, iang in inner_list:
            if not inner_hex_in_outer((ix, iy), iang, (cx, cy), outer_side, outer_angle):
                return False
        # Check pairwise non-overlap (approximate: center distance >= 2 for aligned)
        for i in range(len(inner_list)):
            for j in range(i + 1, len(inner_list)):
                x1, y1, a1 = inner_list[i]
                x2, y2, a2 = inner_list[j]
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                # For same orientation, need dist >= 2
                # For different orientations, minimum distance can be less
                # Conservative: require dist >= 1.9 for safety
                if dist < 1.9:
                    return False
        return True
    
    def min_side_for_packing(inner_list, outer_center, outer_angle):
        """Binary search for minimum outer side that contains all inner hexagons."""
        cx, cy = outer_center
        # Find max required side
        lo = 0.0
        hi = 50.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if all(inner_hex_in_outer((ix, iy), iang, (cx, cy), mid, outer_angle) 
                   for ix, iy, iang in inner_list):
                hi = mid
            else:
                lo = mid
        return hi + 1e-10
    
    # Try different outer angles and centers
    best_side = float('inf')
    best_result = None
    
    # Try center at origin first
    outer_center = (0.0, 0.0)
    
    # Try several outer angles
    for outer_angle in [0.0, 30.0, 15.0, 45.0, 60.0]:
        if time.time() > deadline - 5:
            break
        side = min_side_for_packing(inner, outer_center, outer_angle)
        if check_valid(inner, outer_center, side, outer_angle):
            if side < best_side:
                best_side = side
                best_result = (inner, outer_center, side, outer_angle)
    
    # Also try shifting the inner cluster to center it better
    # Compute centroid of inner hexagons
    if inner:
        cx_mean = sum(p[0] for p in inner) / len(inner)
        cy_mean = sum(p[1] for p in inner) / len(inner)
        # Shift inner so centroid is at origin
        shifted_inner = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in inner]
        
        for outer_angle in [0.0, 30.0, 15.0, 45.0, 60.0]:
            if time.time() > deadline - 5:
                break
            side = min_side_for_packing(shifted_inner, (0.0, 0.0), outer_angle)
            if check_valid(shifted_inner, (0.0, 0.0), side, outer_angle):
                if side < best_side:
                    best_side = side
                    best_result = (shifted_inner, (0.0, 0.0), side, outer_angle)
    
    if best_result is None:
        # Fallback: use the original line packing but with better side
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END
