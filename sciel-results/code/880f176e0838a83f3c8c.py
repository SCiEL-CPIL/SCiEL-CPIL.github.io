import numpy as np
import time

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return cx + side * np.cos(angles), cy + side * np.sin(angles)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = cos_a * dx - sin_a * dy
    ly = sin_a * dx + cos_a * dy
    
    verts_x = side * np.cos(np.arange(6) * np.pi / 3)
    verts_y = side * np.sin(np.arange(6) * np.pi / 3)
    
    for i in range(6):
        x1, y1 = verts_x[i], verts_y[i]
        x2, y2 = verts_x[(i+1)%6], verts_y[(i+1)%6]
        ex = x2 - x1
        ey = y2 - y1
        px2 = lx - x1
        py2 = ly - y1
        cross = ex * py2 - ey * px2
        if cross < -1e-10:
            return False
    return True

def _check_valid(inner, center, side, angle):
    cx, cy = center
    for ix, iy, iang in inner:
        ixv, iyv = _hex_vertices(ix, iy, 1.0, iang)
        for vx, vy in zip(ixv, iyv):
            if not _point_in_hexagon(vx, vy, cx, cy, side, angle):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    sqrt3 = np.sqrt(3)
    inner = []
    for k in range(-1, 2):
        inner.append([k * sqrt3, 1.5, 0.0])
    for k in range(-2, 3):
        inner.append([k * sqrt3, 0.0, 0.0])
    for k in range(-1, 2):
        inner.append([k * sqrt3, -1.5, 0.0])
    
    best_side = 40.0
    best_angle = 0.0
    
    for angle in range(0, 180, 2):
        lo, hi = 3.0, 10.0
        for _ in range(25):
            mid = (lo + hi) / 2
            if _check_valid(inner, (0.0, 0.0), mid, float(angle)):
                hi = mid
            else:
                lo = mid
        if hi < best_side:
            best_side = hi
            best_angle = float(angle)
        
        if time.time() > deadline - 15.0:
            break
    
    return inner, (0.0, 0.0), best_side, best_angle
