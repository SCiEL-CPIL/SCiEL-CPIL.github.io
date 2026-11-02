"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Generate hexagon vertices for a unit hexagon at origin with given angle
    def hex_vertices(cx, cy, angle_deg, side=1.0):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + math.radians(60 * i)
            verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
        return verts
    
    # Check if a point is inside a regular hexagon (center, side, angle)
    def point_in_hex(px, py, hx, hy, side, angle_deg):
        # Transform point to hexagon's local frame
        dx = px - hx
        dy = py - hy
        angle_rad = math.radians(-angle_deg)
        lx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
        ly = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
        # In local frame, hexagon has vertices at angles 0,60,...,300
        # The hexagon contains points where max over edges of (point dot normal) <= inradius
        inradius = side * math.sqrt(3) / 2
        for i in range(6):
            edge_angle = math.radians(60 * i + 30)  # normals to edges
            nx = math.cos(edge_angle)
            ny = math.sin(edge_angle)
            if lx * nx + ly * ny > inradius + 1e-9:
                return False
        return True
    
    # Check if all inner hexagons are inside the outer hexagon
    def all_inside(inner, outer_center, outer_side, outer_angle):
        ox, oy = outer_center
        for (ix, iy, iangle) in inner:
            verts = hex_vertices(ix, iy, iangle)
            for (vx, vy) in verts:
                if not point_in_hex(vx, vy, ox, oy, outer_side, outer_angle):
                    return False
        return True
    
    # Check if inner hexagons are disjoint (no overlap)
    def hexagons_disjoint(inner, tol=1e-6):
        # Two regular hexagons of side 1 are disjoint if distance between centers > 2
        # (when aligned) but for arbitrary rotations, we need to check edges
        # Simplified: check if any vertex of one is inside the other
        for i in range(len(inner)):
            for j in range(i+1, len(inner)):
                ix, iy, ia = inner[i]
                jx, jy, ja = inner[j]
                # Quick reject: center distance
                d = math.sqrt((ix-jx)**2 + (iy-jy)**2)
                if d > 2.0 + tol:
                    continue
                # Check if any vertex of hex i is inside hex j, or vice versa
                vi = hex_vertices(ix, iy, ia)
                vj = hex_vertices(jx, jy, ja)
                for (vx, vy) in vi:
                    if point_in_hex(vx, vy, jx, jy, 1.0, ja):
                        return False
                for (vx, vy) in vj:
                    if point_in_hex(vx, vy, ix, iy, 1.0, ia):
                        return False
                # Also check edge intersections for robustness
                # For now, vertex-in-hex check should suffice for non-overlapping
        return True
    
    # Find minimum outer side length for a given inner arrangement and outer center/angle
    def find_min_side(inner, outer_center, outer_angle):
        # Binary search on side length
        ox, oy = outer_center
        lo = 1.0
        hi = 20.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if all_inside(inner, (ox, oy), mid, outer_angle):
                hi = mid
            else:
                lo = mid
        return hi
    
    # Try honeycomb-like arrangements
    best_score = float('inf')
    best_result = None
    
    # Strategy 1: Grid arrangements with various rotations
    # For flat-top hexagons in honeycomb:
    # Horizontal spacing: 1.5, Vertical spacing: sqrt(3)
    # Let's try 3 rows x 4 cols
    
    sqrt3 = math.sqrt(3)
    
    # Generate several candidate arrangements
    arrangements = []
    
    # Arrangement A: 3x4 honeycomb, all same orientation
    for row in range(3):
        for col in range(4):
            x = col * 1.5
            y = row * sqrt3
            if row % 2 == 1:
                x += 0.75
            arrangements.append([x, y, 0.0])
    
    # Arrangement B: 4x3 honeycomb
    for row in range(4):
        for col in range(3):
            x = col * 1.5
            y = row * sqrt3
            if row % 2 == 1:
                x += 0.75
            arrangements.append([x, y, 0.0])
    
    # Arrangement C: 2x6 honeycomb
    for row in range(2):
        for col in range(6):
            x = col * 1.5
            y = row * sqrt3
            if row % 2 == 1:
                x += 0.75
            arrangements.append([x, y, 0.0])
    
    # Arrangement D: 6x2 honeycomb
    for row in range(6):
        for col in range(2):
            x = col * 1.5
            y = row * sqrt3
            if row % 2 == 1:
                x += 0.75
            arrangements.append([x, y, 0.0])
    
    # Arrangement E: 3x4 with 30-degree rotation
    for row in range(3):
        for col in range(4):
            x = col * 1.5
            y = row * sqrt3
            if row % 2 == 1:
                x += 0.75
            arrangements.append([x, y, 30.0])
    
    # For each arrangement, center it and try to find optimal outer hexagon
    for arr in arrangements:
        if time.time() > deadline:
            break
        
        # Center the arrangement
        xs = [a[0] for a in arr]
        ys = [a[1] for a in arr]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        centered_arr = [[x - cx, y - cy, ang] for (x, y, ang) in arr]
        
        # Check if disjoint
        if not hexagons_disjoint(centered_arr):
            continue
        
        # Try different outer hexagon angles
        for outer_angle in [0.0, 30.0, 15.0, 45.0, 60.0]:
            if time.time() > deadline:
                break
            
            side = find_min_side(centered_arr, (0.0, 0.0), outer_angle)
            
            if side < best_score:
                best_score = side
                best_result = (centered_arr, (0.0, 0.0), side, outer_angle)
    
    # If we found a valid result, return it
    if best_result is not None:
        return best_result
    
    # Fallback: return the seed arrangement
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END

