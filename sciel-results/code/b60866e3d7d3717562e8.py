"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    angle_rad = math.radians(angle_deg)
    return [(cx + side * math.cos(angle_rad + i * math.pi / 3),
             cy + side * math.sin(angle_rad + i * math.pi / 3)) for i in range(6)]

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    dx, dy = px - cx, py - cy
    angle_rad = math.radians(-angle_deg)
    lx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
    ly = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
    for i in range(6):
        a = i * math.pi / 3
        if lx * math.cos(a) + ly * math.sin(a) > side * math.sqrt(3) / 2 + 1e-10:
            return False
    return True

def hexagon_overlap(v1, v2):
    for verts in [v1, v2]:
        for i in range(6):
            ax1, ay1 = verts[i]
            ax2, ay2 = verts[(i+1) % 6]
            ex, ey = ax2 - ax1, ay2 - ay1
            nx, ny = -ey, ex
            nlen = math.sqrt(nx*nx + ny*ny)
            if nlen < 1e-12:
                continue
            nx, ny = nx/nlen, ny/nlen
            min1 = min(vx*nx + vy*ny for vx, vy in v1)
            max1 = max(vx*nx + vy*ny for vx, vy in v1)
            min2 = min(vx*nx + vy*ny for vx, vy in v2)
            max2 = max(vx*nx + vy*ny for vx, vy in v2)
            if max1 < min2 - 1e-10 or max2 < min1 - 1e-10:
                return False
    return True

def bounding_hexagon_side(hexagons, outer_center, outer_angle):
    cx, cy = outer_center
    angle_rad = math.radians(outer_angle)
    max_R = 0
    for hx, hy, hangle in hexagons:
        for vx, vy in hex_vertices(hx, hy, 1.0, hangle):
            dx, dy = vx - cx, vy - cy
            rx = dx * math.cos(-angle_rad) - dy * math.sin(-angle_rad)
            ry = dx * math.sin(-angle_rad) + dy * math.cos(-angle_rad)
            for i in range(6):
                a = i * math.pi / 3
                proj = rx * math.cos(a) + ry * math.sin(a)
                if proj > 0:
                    R = proj / (math.sqrt(3) / 2)
                    if R > max_R:
                        max_R = R
    return max_R

def is_valid(inner, oc, oside, oangle):
    cx, cy = oc
    for i in range(len(inner)):
        vi = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for j in range(i+1, len(inner)):
            vj = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if hexagon_overlap(vi, vj):
                return False
    for hx, hy, ha in inner:
        for vx, vy in hex_vertices(hx, hy, 1.0, ha):
            if not point_in_hexagon(vx, vy, cx, cy, oside, oangle):
                return False
    return True

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140
    
    d = math.sqrt(3)
    pos = [(0,0)]
    for i in range(6):
        a = i * math.pi / 3
        pos.append((d*math.cos(a), d*math.sin(a)))
    for i in range(6):
        a = i * math.pi / 3
        pos.append((2*d*math.cos(a), 2*d*math.sin(a)))
    pos = pos[:11]
    
    inner = [[p[0], p[1], 0.0] for p in pos]
    oc = (0.0, 0.0)
    oangle = 0.0
    oside = bounding_hexagon_side(inner, oc, oangle)
    
    best = [row[:] for row in inner]
    best_side = oside
    
    def eval_cfg(cfg):
        s = bounding_hexagon_side(cfg, oc, oangle)
        if is_valid(cfg, oc, s, oangle):
            return s
        return float('inf')
    
    step = 0.15
    for it in range(8000):
        if time.time() > deadline:
            break
        idx = np.random.randint(0, N)
        ox, oy, oa = inner[idx]
        for dx in [-step, 0, step]:
            for dy in [-step, 0, step]:
                if dx==0 and dy==0: continue
                inner[idx] = [ox+dx, oy+dy, oa]
                s = eval_cfg(inner)
                if s < best_side - 1e-12:
                    best_side = s
                    best = [r[:] for r in inner]
                    break
                else:
                    inner[idx] = [ox, oy, oa]
            else:
                continue
            break
        if it % 400 == 399:
            step *= 0.85
    
    return best, oc, best_side, oangle
# EVOLVE-BLOCK-END

