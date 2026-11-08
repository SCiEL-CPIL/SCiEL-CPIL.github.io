"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
import random

N = 11

# EVOLVE-BLOCK-START
def _hex_verts(cx, cy, ang):
    """6 vertices of unit hexagon."""
    a = math.radians(ang)
    return [(cx + math.cos(a + i*math.pi/3), cy + math.sin(a + i*math.pi/3)) for i in range(6)]

def _inside_hex(px, py, cx, cy, ang):
    """Is point (px,py) inside unit hex at (cx,cy) with angle ang?"""
    a = math.radians(ang)
    dx, dy = px-cx, py-cy
    ca, sa = math.cos(-a), math.sin(-a)
    lx, ly = ca*dx - sa*dy, sa*dx + ca*dy
    # Unit hex (pointy-top): check 6 half-planes
    for i in range(6):
        t = math.radians(30 + i*60)
        nx, ny = math.cos(t), math.sin(t)
        if lx*nx + ly*ny > math.sqrt(3)/2 + 1e-9:
            return False
    return True

def _min_dist_hex(cx1, cy1, a1, cx2, cy2, a2):
    """Min distance between two unit hexagons."""
    v1 = _hex_verts(cx1, cy1, a1)
    v2 = _hex_verts(cx2, cy2, a2)
    # Check vertex-in-hex
    for vx, vy in v1:
        if _inside_hex(vx, vy, cx2, cy2, a2):
            return 0.0
    for vx, vy in v2:
        if _inside_hex(vx, vy, cx1, cy1, a1):
            return 0.0
    # Min segment distance
    min_d = float('inf')
    for i in range(6):
        ax, ay = v1[i]
        bx, by = v1[(i+1)%6]
        for j in range(6):
            cx, cy = v2[j]
            dx, dy = v2[(j+1)%6]
            # Segment-segment distance
            # Vector from a to c
            acx, acy = cx-ax, cy-ay
            abx, aby = bx-ax, by-ay
            adx, ady = dx-cx, dy-cy
            # Denominators
            a2v = abx*abx + aby*aby
            c2v = adx*adx + ady*ady
            b2v = abx*adx + aby*ady
            if a2v < 1e-12 and c2v < 1e-12:
                d = math.hypot(acx, acy)
            elif a2v < 1e-12:
                t = max(0, min(1, (acx*adx+acy*ady)/c2v))
                d = math.hypot(cx+adx*t-ax, cy+ady*t-ay)
            elif c2v < 1e-12:
                t = max(0, min(1, (acx*abx+acy*aby)/a2v))
                d = math.hypot(ax+abx*t-cx, ay+aby*t-cy)
            else:
                det = a2v*c2v - b2v*b2v
                if det < 1e-12:
                    d = min(math.hypot(acx,acy), math.hypot(bx-cx,by-cy), 
                           math.hypot(ax-dx,ay-dy), math.hypot(bx-dx,by-dy))
                else:
                    s_val = (b2v*acy - c2v*acx) / det
                    t_val = (a2v*acy - b2v*acx) / det
                    if s_val < 0:
                        s_val = 0
                        t_val = max(0, min(1, -b2v/c2v))
                    elif s_val > 1:
                        s_val = 1
                        t_val = max(0, min(1, (b2v-c2v)/c2v))
                    elif t_val < 0:
                        t_val = 0
                        s_val = max(0, min(1, -acx/a2v))
                    elif t_val > 1:
                        t_val = 1
                        s_val = max(0, min(1, (b2v-a2v)/a2v))
                    px1, py1 = ax+abx*s_val, ay+aby*s_val
                    px2, py2 = cx+adx*t_val, cy+ady*t_val
                    d = math.hypot(px1-px2, py1-py2)
            if d < min_d:
                min_d = d
    return min_d

def _outer_size(pos, angs):
    """Min outer hex side length (centered at origin, rot=0)."""
    max_ap = 0.0
    for (cx, cy), ang in zip(pos, angs):
        for vx, vy in _hex_verts(cx, cy, ang):
            for i in range(6):
                t = math.radians(30 + i*60)
                nx, ny = math.cos(t), math.sin(t)
                ap = vx*nx + vy*ny
                if ap > max_ap:
                    max_ap = ap
    return max_ap / (math.sqrt(3)/2)

def _overlap_penalty(pos, angs):
    pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _min_dist_hex(pos[i][0], pos[i][1], angs[i], pos[j][0], pos[j][1], angs[j])
            if d < 0.05:
                pen += (0.05 - d) * 50
    return pen

def construct():
    start = time.time()
    deadline = start + 105.0  # Safety margin
    
    # Initial: hex grid pattern 3-4-3-1
    s = math.sqrt(3)
    vs = s * 1.5
    pos = []
    angs = []
    for k in range(-1, 2):
        pos.append([k*s, 0.0]); angs.append(0.0)
    for k in range(-1, 3):
        pos.append([k*s - s/2, vs]); angs.append(0.0)
    for k in range(-1, 2):
        pos.append([k*s, 2*vs]); angs.append(0.0)
    pos.append([0.0, 3*vs]); angs.append(0.0)
    
    # Center
    xs = [p[0] for p in pos]; ys = [p[1] for p in pos]
    cx0 = (min(xs)+max(xs))/2; cy0 = (min(ys)+max(ys))/2
    pos = [[x-cx0, y-cy0] for x,y in pos]
    
    best_pos = [list(p) for p in pos]
    best_ang = list(angs)
    best_size = _outer_size(pos, angs)
    
    T = 0.3
    T_min = 0.0005
    cooling = 0.998
    rng = random.Random(123)
    
    cur_pos = [list(p) for p in pos]
    cur_ang = list(angs)
    cur_score = _outer_size(cur_pos, cur_ang) + _overlap_penalty(cur_pos, cur_ang) * 10
    
    while time.time() < deadline:
        i = rng.randint(0, N-1)
        mt = rng.random()
        if mt < 0.5:
            dx, dy = rng.gauss(0,0.2), rng.gauss(0,0.2)
            cur_pos[i][0] += dx; cur_pos[i][1] += dy
            revert = lambda: (cur_pos[i].__setitem__(0, cur_pos[i][0]-dx), cur_pos[i].__setitem__(1, cur_pos[i][1]-dy))
        elif mt < 0.8:
            da = rng.gauss(0, 8)
            cur_ang[i] += da
            revert = lambda: cur_ang.__setitem__(i, cur_ang[i]-da)
        else:
            dx, dy = rng.gauss(0,0.1), rng.gauss(0,0.1)
            da = rng.gauss(0, 5)
            cur_pos[i][0] += dx; cur_pos[i][1] += dy; cur_ang[i] += da
            revert = lambda: (cur_pos[i].__setitem__(0, cur_pos[i][0]-dx), cur_pos[i].__setitem__(1, cur_pos[i][1]-dy), cur_ang.__setitem__(i, cur_ang[i]-da))
        
        new_score = _outer_size(cur_pos, cur_ang) + _overlap_penalty(cur_pos, cur_ang) * 10
        
        if new_score < cur_score or rng.random() < math.exp(-(new_score - cur_score)/max(T, 1e-10)):
            cur_score = new_score
            if new_score < best_size and _overlap_penalty(cur_pos, cur_ang) < 0.01:
                best_pos = [list(p) for p in cur_pos]
                best_ang = list(cur_ang)
                best_size = _outer_size(best_pos, best_ang)
        else:
            revert()
        
        T = max(T * cooling, T_min)
    
    inner = [[p[0], p[1], a] for p, a in zip(best_pos, best_ang)]
    return inner, (0.0, 0.0), best_size, 0.0
# EVOLVE-BLOCK-END
