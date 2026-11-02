"""Optimized hexagon packing: 11 unit hexagons in a hexagonal cluster with precise optimization."""
import math
import time
import numpy as np

N = 11


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    # Define the 11 hexagon positions in a compact hexagonal cluster.
    # This is a 1-2-3-2-1 pattern (hexagonal number H_3 = 19, but we use 11).
    # Actually, let's use a more compact arrangement.
    
    # A good starting configuration: center + 6 neighbors + 4 from ring 2
    # Use axial coordinates for hexagonal grid
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # Center + ring 1 (6 hexagons) + 4 selected from ring 2
    # For 11 hexagons, we want a compact shape. Let's use:
    # Row 0: 1 hex (center)
    # Row 1: 3 hexes
    # Row 2: 3 hexes  
    # Row 3: 3 hexes
    # Row 4: 1 hex
    # That's 11 total, forming a hexagonal shape
    
    # Actually, the most compact 11-hexagon packing is known to be
    # a hexagonal cluster. Let me use the standard 1-2-3-2-1 hexagon:
    # That's actually 9 hexagons. For 11, we need to extend.
    
    # Let me use a different approach: start with 7 (center+6) and add 4 more
    # in a way that minimizes the bounding hexagon.
    
    # Positions for 11 hexagons in a compact arrangement:
    # Using axial coordinates (q, r) where x = 1.5q, y = sqrt(3)*(r + q/2)
    
    # A good compact 11-hexagon shape:
    axial_positions = [
        (0, 0),      # center
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),  # ring 1
        (2, -1), (1, -2), (-1, -1), (-2, 1)  # 4 from ring 2
    ]
    
    # Convert to Cartesian
    positions = [axial_to_xy(q, r) for q, r in axial_positions]
    
    # Compute centroid
    cx0 = sum(x for x, y in positions) / N
    cy0 = sum(y for x, y in positions) / N
    
    # Shift so centroid is at origin
    positions = [(x - cx0, y - cy0) for x, y in positions]
    
    # Now optimize the outer hexagon (center, side, angle)
    # The outer hexagon is centered at (0,0) with some rotation
    
    def hex_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon."""
        verts = []
        for k in range(6):
            a = math.radians(angle_deg + 60 * k)
            verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
        return verts
    
    def point_in_hex(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside hexagon."""
        dx = px - cx
        dy = py - cy
        a = -math.radians(angle_deg)
        lx = dx * math.cos(a) - dy * math.sin(a)
        ly = dx * math.sin(a) + dy * math.cos(a)
        apothem = side * sqrt3 / 2
        for k in range(6):
            a_norm = math.radians(30 + 60 * k)
            nx = math.cos(a_norm)
            ny = math.sin(a_norm)
            if lx * nx + ly * ny > apothem + 1e-9:
                return False
        return True
    
    def min_outer_side_for_config(positions, ox, oy, oangle):
        """Find minimum side length of outer hexagon that contains all inner hexagons."""
        max_val = 0.0
        a = -math.radians(oangle)
        cos_a = math.cos(a)
        sin_a = math.sin(a)
        
        for cx, cy in positions:
            # For each inner hexagon, check all 6 vertices
            for k in range(6):
                va = math.radians(k * 60)
                vx = cx + math.cos(va)
                vy = cy + math.sin(va)
                
                dx = vx - ox
                dy = vy - oy
                lx = dx * cos_a - dy * sin_a
                ly = dx * sin_a + dy * cos_a
                
                for m in range(6):
                    theta = math.radians(30 + 60 * m)
                    proj = lx * math.cos(theta) + ly * math.sin(theta)
                    if proj > max_val:
                        max_val = proj
        
        return max_val * 2.0 / sqrt3
    
    # Optimize outer hexagon angle and center
    best_side = float('inf')
    best_cx = 0.0
    best_cy = 0.0
    best_angle = 0.0
    
    # Try a range of angles
    for oangle in np.linspace(0, 60, 121):
        # For each angle, try a small grid of centers
        for dx in np.linspace(-0.5, 0.5, 11):
            for dy in np.linspace(-0.5, 0.5, 11):
                side = min_outer_side_for_config(positions, dx, dy, oangle)
                if side < best_side:
                    best_side = side
                    best_cx = dx
                    best_cy = dy
                    best_angle = oangle
    
    # Fine-tune with local optimization
    cur_cx, cur_cy, cur_ang, cur_side = best_cx, best_cy, best_angle, best_side
    
    for step in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    for da in [-step*10, 0, step*10]:
                        ncx = cur_cx + dx
                        ncy = cur_cy + dy
                        nang = cur_ang + da
                        s = min_outer_side_for_config(positions, ncx, ncy, nang)
                        if s < cur_side - 1e-12:
                            cur_cx, cur_cy, cur_ang, cur_side = ncx, ncy, nang, s
                            improved = True
    
    # Now try to improve by moving inner hexagons slightly
    # This is a joint optimization: move inner hexagons and outer hexagon together
    
    cur_positions = list(positions)
    
    for step in [0.02, 0.01, 0.005, 0.002]:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(N):
                px, py = cur_positions[i]
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        new_pos = list(cur_positions)
                        new_pos[i] = (px + dx, py + dy)
                        
                        # Check disjointness
                        valid = True
                        for j in range(N):
                            if j == i:
                                continue
                            # Check if hexagons i and j overlap
                            # Two unit hexagons overlap if their centers are too close
                            # For hexagons with same orientation, min distance is 2*apothem = sqrt(3)
                            dxj = new_pos[i][0] - new_pos[j][0]
                            dyj = new_pos[i][1] - new_pos[j][1]
                            dist = math.sqrt(dxj*dxj + dyj*dyj)
                            if dist < sqrt3 - 1e-9:
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        s = min_outer_side_for_config(new_pos, cur_cx, cur_cy, cur_ang)
                        if s < cur_side - 1e-10:
                            cur_positions = new_pos
                            cur_side = s
                            improved = True
    
    # Final polish
    if time.time() < deadline:
        for step in [0.001, 0.0005, 0.0001]:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for dx in [-step, step]:
                    for dy in [-step, step]:
                        for da in [-step*5, step*5]:
                            ncx = cur_cx + dx
                            ncy = cur_cy + dy
                            nang = cur_ang + da
                            s = min_outer_side_for_config(cur_positions, ncx, ncy, nang)
                            if s < cur_side - 1e-13:
                                cur_cx, cur_cy, cur_ang, cur_side = ncx, ncy, nang, s
                                improved = True
    
    # Build final inner hexagons (all with 0 degree rotation, aligned with outer)
    inner_hexes = [(x, y, 0.0) for x, y in cur_positions]
    
    # Verify validity: all inner hexagons must be inside outer
    outer_cx, outer_cy, outer_side, outer_angle = cur_cx, cur_cy, cur_side, cur_ang
    
    # Check each inner hexagon's vertices are inside outer
    valid = True
    for ix, iy, iang in inner_hexes:
        for k in range(6):
            va = math.radians(iang + 60 * k)
            vx = ix + math.cos(va)
            vy = iy + math.sin(va)
            if not point_in_hex(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Binary search for minimum valid side
        lo, hi = cur_side, cur_side * 1.05
        for _ in range(40):
            mid = (lo + hi) / 2
            ok = True
            for ix, iy, iang in inner_hexes:
                for k in range(6):
                    va = math.radians(iang + 60 * k)
                    vx = ix + math.cos(va)
                    vy = iy + math.sin(va)
                    if not point_in_hex(vx, vy, outer_cx, outer_cy, mid, outer_angle):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                hi = mid
            else:
                lo = mid
        cur_side = hi
    
    # Format output
    inner = [[x, y, ang] for x, y, ang in inner_hexes]
    center = (cur_cx, cur_cy)
    
    return inner, center, cur_side, cur_ang
