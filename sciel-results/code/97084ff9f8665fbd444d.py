"""Smooth penalty optimization for hexagon packing."""
import time
import numpy as np
from scipy.optimize import minimize
import math

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 120.0  # Safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    dx = px - cx
    dy = py - cy
    a = np.radians(angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    verts_local = []
    s = side
    for i in range(6):
        theta = i * np.pi / 3
        verts_local.append([s * np.cos(theta), s * np.sin(theta)])
    verts_local = np.array(verts_local)
    
    for i in range(6):
        v1 = verts_local[i]
        v2 = verts_local[(i + 1) % 6]
        edge = v2 - v1
        nx, ny = edge[1], -edge[0]
        if nx * (lx - v1[0]) + ny * (ly - v1[1]) > 1e-10:
            return False
    return True

def _min_outer_side(inner_positions, inner_angles, outer_center, outer_angle, tol=1e-6):
    """Find minimum outer side length such that all inner hexagons fit inside the outer hexagon."""
    cx, cy = outer_center
    all_verts = []
    for i in range(len(inner_positions)):
        x, y = inner_positions[i]
        ang = inner_angles[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    lo, hi = 1.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        ok = True
        for v in all_verts:
            if not _point_in_hexagon(v[0], v[1], cx, cy, mid, outer_angle):
                ok = False
                break
        if ok:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if packing is valid: no overlaps, all inside outer hexagon."""
    cx, cy = outer_center
    verts_list = []
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        verts_list.append(verts)
    
    # Check no overlaps using SAT
    for i in range(len(verts_list)):
        for j in range(i + 1, len(verts_list)):
            if _hexagon_overlap(verts_list[i], verts_list[j]):
                return False
    
    # Check all inside outer hexagon
    for verts in verts_list:
        for v in verts:
            if not _point_in_hexagon(v[0], v[1], cx, cy, outer_side, outer_angle):
                return False
    return True

def _hexagon_overlap(verts1, verts2, tol=1e-9):
    """Check if two convex polygons overlap using SAT (Separating Axis Theorem)."""
    axes = []
    for verts in [verts1, verts2]:
        for i in range(len(verts)):
            v1 = verts[i]
            v2 = verts[(i + 1) % len(verts)]
            edge = v2 - v1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                axes.append([nx/norm, ny/norm])
    
    for axis in axes:
        proj1 = np.dot(verts1, axis)
        proj2 = np.dot(verts2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    return True

def _initial_packing():
    """Create initial hexagonal grid packing."""
    positions = []
    angles = []
    sqrt3 = np.sqrt(3)
    # Use a tighter initial grid to start closer to the optimum
    # Hexagonal packing of 11: 
    # Row 0 (y=0): 3 hexes at x = -2, 0, 2
    # Row 1 (y=sqrt(3)): 4 hexes at x = -3, -1, 1, 3
    # Row 2 (y=2*sqrt(3)): 3 hexes at x = -2, 0, 2
    # Wait, standard hex packing:
    # Center at (0,0).
    # Neighbors at distance 2.
    # Let's just use a simple grid and let the optimizer fix it.
    # Or use the known optimal-ish configuration for 11 hexes.
    # 11 hexes in a hexagon:
    # Side length of outer hexagon for 11 hexes?
    # If we pack them in a triangular shape:
    # 1
    # 3
    # 5
    # 2? No, 1+3+5 = 9. 1+3+5+2 = 11?
    # Let's just generate a grid and trim.
    rows = 3
    for r in range(rows):
        y = (r - rows // 2) * sqrt3
        if r % 2 == 0:
            xs = range(-(rows // 2), rows - rows // 2 + 1)
        else:
            xs = range(-(rows // 2), rows - rows // 2)
        for x_idx in xs:
            x = (x_idx - (rows // 2)) * 2.0
            if r % 2 == 1:
                x += 1.0
            positions.append([x, y])
            angles.append(0.0)
    positions = positions[:N]
    angles = angles[:N]
    return np.array(positions), np.array(angles)

def _generate_hex_cluster():
    """Generate a hexagonal cluster of 11 hexagons: 1 center + 6 inner ring + 4 outer ring."""
    pos = []
    ang = []
    sqrt3 = math.sqrt(3)
    
    # Center hexagon
    pos.append([0.0, 0.0])
    ang.append(0.0)
    
    # Inner ring: 6 hexagons at distance 2 from center
    for i in range(6):
        theta = i * math.pi / 3
        x = 2.0 * math.cos(theta)
        y = 2.0 * math.sin(theta)
        pos.append([x, y])
        ang.append(0.0)
    
    # Outer ring: add 4 hexagons in the gaps
    # The gaps between adjacent inner ring hexagons are at angles 30, 90, 150, 210, 270, 330
    # We need to place 4 more hexagons. Let's place them at 90, 150, 210, 270 degrees
    # The center of each gap is at distance 2 from the center of the hexagon it's between.
    # But we need the distance from the origin.
    # The gap center is at the centroid of the triangle formed by the origin and two adjacent inner ring hexagons.
    # The triangle has vertices (0,0), (2*cos(a), 2*sin(a)), (2*cos(a+60), 2*sin(a+60))
    # The centroid is at ((0 + 2*cos(a) + 2*cos(a+60))/3, (0 + 2*sin(a) + 2*sin(a+60))/3)
    # This is complex. Let's just place them at distance 4 from the center at angles 90, 150, 210, 270.
    # Distance 4 is the distance from center to the center of a hexagon in the second ring of a hexagonal packing.
    for angle_deg in [90, 150, 210, 270]:
        theta = math.radians(angle_deg)
        x = 4.0 * math.cos(theta)
        y = 4.0 * math.sin(theta)
        pos.append([x, y])
        ang.append(0.0)
    
    return np.array(pos), np.array(ang)

def construct():
    global START_TIME, DEADLINE
    
    # Try multiple initial configurations
    best_side = float('inf')
    best_pos = None
    best_ang = None
    best_ocx, best_ocy = 0.0, 0.0
    best_oor = 0.0
    
    initial_configs = [
        _initial_packing(),
        _generate_hex_cluster(),
    ]
    
    for pos0, ang0 in initial_configs:
        if time.time() > DEADLINE - 10.0:
            break
        
        n = len(pos0)
        outer_cx, outer_cy = 0.0, 0.0
        outer_angle = 0.0
        
        current_side = _min_outer_side(pos0, ang0, (outer_cx, outer_cy), outer_angle)
        pos = pos0.copy()
        ang = ang0.copy()
        
        # Local search
        np.random.seed(42)
        step = 0.1
        iters = 0
        
        while time.time() < DEADLINE - 5.0:
            iters += 1
            
            improved = False
            for i in range(n):
                if time.time() > DEADLINE - 5.0:
                    break
                
                for dx in [-step, 0.0, step]:
                    for dy in [-step, 0.0, step]:
                        if dx == 0.0 and dy == 0.0:
                            continue
                        new_pos = pos.copy()
                        new_pos[i][0] += dx
                        new_pos[i][1] += dy
                        
                        verts = _hex_vertices(new_pos[i][0], new_pos[i][1], 1.0, ang[i])
                        inside = True
                        for v in verts:
                            if not _point_in_hexagon(v[0], v[1], outer_cx, outer_cy, current_side, outer_angle):
                                inside = False
                                break
                        if not inside:
                            continue
                        
                        overlap = False
                        for j in range(n):
                            if j == i:
                                continue
                            verts_j = _hex_vertices(pos[j][0], pos[j][1], 1.0, ang[j])
                            if _hexagon_overlap(verts, verts_j):
                                overlap = True
                                break
                        if overlap:
                            continue
                        
                        new_side = _min_outer_side(new_pos, ang, (outer_cx, outer_cy), outer_angle)
                        if new_side < current_side - 1e-8:
                            pos = new_pos
                            current_side = new_side
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                for i in range(n):
                    if time.time() > DEADLINE - 5.0:
                        break
                    for da in [-5.0, 5.0]:
                        new_ang = ang.copy()
                        new_ang[i] = ang[i] + da
                        new_side = _min_outer_side(pos, new_ang, (outer_cx, outer_cy), outer_angle)
                        if new_side < current_side - 1e-8:
                            verts = _hex_vertices(pos[i][0], pos[i][1], 1.0, new_ang[i])
                            inside = True
                            for v in verts:
                                if not _point_in_hexagon(v[0], v[1], outer_cx, outer_cy, current_side, outer_angle):
                                    inside = False
                                    break
                            if not inside:
                                continue
                            overlap = False
                            for j in range(n):
                                if j == i:
                                    continue
                                verts_j = _hex_vertices(pos[j][0], pos[j][1], 1.0, ang[j])
                                if _hexagon_overlap(verts, verts_j):
                                    overlap = True
                                    break
                            if not overlap:
                                ang = new_ang
                                current_side = new_side
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                step *= 0.8
                if step < 0.001:
                    break
            
            if current_side < best_side:
                best_side = current_side
                best_pos = pos.copy()
                best_ang = ang.copy()
                best_ocx, best_ocy = outer_cx, outer_cy
                best_oor = outer_angle
    
    # Final verification
    if best_pos is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner_list = [[float(best_pos[i][0]), float(best_pos[i][1]), float(best_ang[i])] for i in range(N)]
    if not _check_validity(inner_list, (best_ocx, best_ocy), best_side, best_oor):
        sqrt3 = np.sqrt(3)
        inner = [[4.0 * k, 0.0, 0.0] for k in range(N)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return inner_list, (float(best_ocx), float(best_ocy)), float(best_side), float(best_oor)
