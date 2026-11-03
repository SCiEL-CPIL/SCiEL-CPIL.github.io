"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random

N = 12
TIME_LIMIT = 118.0  # Safety margin below 120s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    """Get edge normals (perpendicular to edges) for SAT."""
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        # Edge direction
        dx, dy = x2 - x1, y2 - y1
        # Normal
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _separating_axis_check(vertsA, vertsB):
    """Check if two convex polygons overlap using SAT. Return True if they overlap (not disjoint)."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False  # Found separating axis -> no overlap
    return True  # No separating axis -> overlap

def _outer_hex_side(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    # 6 outward normals of the outer hexagon
    # The outer hexagon is centered at 'center' and rotated by 'angle_deg'
    # Its sides are perpendicular to these normals
    normals = []
    for k in range(6):
        # The normals of the outer hexagon sides
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _is_valid(centers_angles):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            if _separating_axis_check(polys[i], polys[j]):
                return False
    return True

def _initial_config_3x4():
    """Create a 3x4 grid of hexagons. This is a valid, compact starting point."""
    pts = []
    # Hexagon side length is 1.
    # Horizontal distance between centers in a row: 2 * side * sin(60) ? No.
    # For a regular hexagon of side 1:
    # Width (flat-to-flat) = sqrt(3)
    # Height (point-to-point) = 2
    # If we align them with flat sides horizontal (angle 0):
    # Vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # To pack them in a grid without rotation:
    # x-spacing: 2 * (sqrt(3)/2) = sqrt(3) ? No, adjacent flat sides touch.
    # Distance between centers = 2 * apothem = sqrt(3).
    # y-spacing: 2 * apothem = sqrt(3).
    # Wait, if I place them in a square grid with spacing sqrt(3), they touch at corners?
    # Let's use a standard hexagonal lattice packing.
    # Row 0: y=0. x = 0, sqrt(3), 2*sqrt(3), 3*sqrt(3)
    # Row 1: y=1.5 (height of hex is 2, but vertical offset for hex lattice is 1.5? No.)
    # Height of hex = 2. Apothem = sqrt(3)/2.
    # Vertical distance between rows in a "pointy" top lattice?
    # Let's stick to "flat" top (angle 0).
    # Width = sqrt(3). Height = 2.
    # If I shift row 1 by sqrt(3)/2 in x and 1.5 in y?
    # Let's just use a simple grid with sufficient spacing to guarantee validity initially,
    # then let the optimizer tighten it.
    # Spacing 2.0 in x and y is safe (hex fits in 2x2 box).
    for r in range(3):
        for c in range(4):
            x = c * 2.1
            y = r * 2.1
            pts.append([x, y, 0.0])
    # Center the configuration
    avg_x = sum(p[0] for p in pts) / N
    avg_y = sum(p[1] for p in pts) / N
    for p in pts:
        p[0] -= avg_x
        p[1] -= avg_y
    return pts

def construct():
    t0 = time.time()
    random.seed(42)
    
    best_global_score = float('inf')
    best_global_cfg = None
    best_global_center = (0.0, 0.0)
    best_global_angle = 0.0
    
    # Multi-start: Try a few different initializations and SA runs
    num_restarts = 5
    time_per_restart = (TIME_LIMIT - 2.0) / num_restarts  # Leave 2s for final check/return
    
    for restart in range(num_restarts):
        if time.time() - t0 > TIME_LIMIT - 2.0:
            break
            
        # Initialize configuration
        if restart == 0:
            cfg = _initial_config_3x4()
        else:
            # Jitter the initial config for diversity
            cfg = _initial_config_3x4()
            for p in cfg:
                p[0] += random.gauss(0, 0.5)
                p[1] += random.gauss(0, 0.5)
                p[2] = random.uniform(-30, 30)
        
        center = [0.0, 0.0]
        angle = 0.0
        
        # Check initial validity
        if not _is_valid(cfg):
            # Spread out if invalid
            for p in cfg:
                p[0] *= 1.5
                p[1] *= 1.5
            if not _is_valid(cfg):
                # Force valid by large spacing
                for i, p in enumerate(cfg):
                    p[0] = (i % 4 - 1.5) * 4.0
                    p[1] = (i // 4 - 1.0) * 4.0
        
        cur_score = _outer_hex_side(cfg, tuple(center), angle)
        best_score = cur_score
        best_cfg = [row[:] for row in cfg]
        best_center = center[:]
        best_angle = angle
        
        T = 0.8
        T_end = 0.01
        alpha = 0.999
        
        # Local SA Loop
        while time.time() - t0 < t0 + (restart + 1) * time_per_restart:
            r = random.random()
            if r < 0.6:
                idx = random.randrange(N)
                old_x, old_y = cfg[idx][0], cfg[idx][1]
                cfg[idx][0] += random.gauss(0, 0.3)
                cfg[idx][1] += random.gauss(0, 0.3)
            elif r < 0.8:
                idx = random.randrange(N)
                old_ang = cfg[idx][2]
                cfg[idx][2] += random.gauss(0, 10.0)
            elif r < 0.9:
                old_cx, old_cy = center[0], center[1]
                center[0] += random.gauss(0, 0.2)
                center[1] += random.gauss(0, 0.2)
            else:
                old_angle = angle
                angle += random.gauss(0, 5.0)
            
            if not _is_valid(cfg):
                # Revert
                if r < 0.6:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
                elif r < 0.8:
                    cfg[idx][2] = old_ang
                elif r < 0.9:
                    center[0], center[1] = old_cx, old_cy
                else:
                    angle = old_angle
            else:
                new_score = _outer_hex_side(cfg, tuple(center), angle)
                if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                    cur_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_cfg = [row[:] for row in cfg]
                        best_center = center[:]
                        best_angle = angle
                else:
                    # Revert
                    if r < 0.6:
                        cfg[idx][0], cfg[idx][1] = old_x, old_y
                    elif r < 0.8:
                        cfg[idx][2] = old_ang
                    elif r < 0.9:
                        center[0], center[1] = old_cx, old_cy
                    else:
                        angle = old_angle
            
            T = max(T_end, T * alpha)
            
            # Update global best
            if best_score < best_global_score:
                best_global_score = best_score
                best_global_cfg = best_cfg
                best_global_center = tuple(best_center)
                best_global_angle = best_angle

    # Return the best found
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_global_cfg]
    return inner, best_global_center, best_global_score, best_global_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
