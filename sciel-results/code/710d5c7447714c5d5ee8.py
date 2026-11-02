import numpy as np
import time

N = 11
DEADLINE = time.time() + 140.0
SQRT3 = np.sqrt(3.0)

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    return np.array([[cx + side * np.cos(a + k * np.pi / 3.0),
                      cy + side * np.sin(a + k * np.pi / 3.0)] for k in range(6)])

def _separating_axis(v1, v2):
    for poly in (v1, v2):
        for i in range(6):
            p1, p2 = poly[i], poly[(i+1) % 6]
            ex, ey = p2[0]-p1[0], p2[1]-p1[1]
            nx, ny = -ey, ex
            proj1 = v1[:, 0]*nx + v1[:, 1]*ny
            proj2 = v2[:, 0]*nx + v2[:, 1]*ny
            if proj1.max() < proj2.min() - 1e-9 or proj2.max() < proj1.min() - 1e-9:
                return True
    return False

def _all_disjoint(verts):
    for i in range(N):
        for j in range(i+1, N):
            dx = verts[i][0,0] - verts[j][0,0]
            dy = verts[i][0,1] - verts[j][0,1]
            d2 = dx*dx + dy*dy
            if d2 < (SQRT3 - 1e-6)**2:
                return False
            if d2 > 4.0:
                continue
            if not _separating_axis(verts[i], verts[j]):
                return False
    return True

def _min_outer_side(verts, cx, cy):
    best = float('inf')
    all_pts = np.vstack(verts)
    for oa in range(0, 360, 30):
        a = np.radians(oa)
        max_proj = 0.0
        for k in range(6):
            na = a + np.pi/6 + k * np.pi/3.0
            nx, ny = np.cos(na), np.sin(na)
            proj = (all_pts[:, 0] - cx) * nx + (all_pts[:, 1] - cy) * ny
            m = proj.max()
            if m > max_proj:
                max_proj = m
        side = max_proj / (SQRT3 / 2.0)
        if side < best:
            best = side
    return best

def construct():
    t0 = time.time()
    
    # Initial flower + 4
    pos = [(0.0, 0.0)]
    for k in range(6):
        ang = np.radians(30 + k*60)
        pos.append((SQRT3*np.cos(ang), SQRT3*np.sin(ang)))
    for k in [0, 1, 3, 4]:
        ang = np.radians(30 + k*60)
        pos.append((2*SQRT3*np.cos(ang), 2*SQRT3*np.sin(ang)))
    
    inner = [[p[0], p[1], 0.0] for p in pos]
    cx, cy = 0.0, 0.0
    
    verts = [_hex_vertices(x, y, 1.0, a) for x, y, a in inner]
    if not _all_disjoint(verts):
        for k in range(N):
            ang = k * 2*np.pi/N
            inner[k][0] = 4.0*np.cos(ang)
            inner[k][1] = 4.0*np.sin(ang)
        verts = [_hex_vertices(x, y, 1.0, a) for x, y, a in inner]
    
    side = _min_outer_side(verts, cx, cy)
    best_side = side
    best_inner = [list(x) for x in inner]
    
    rng = np.random.RandomState(42)
    step = 0.05
    
    for it in range(200000):
        if time.time() > DEADLINE:
            break
        idx = rng.randint(N)
        dx = rng.normal(0, step)
        dy = rng.normal(0, step)
        ox, oy = inner[idx][0], inner[idx][1]
        inner[idx][0] += dx
        inner[idx][1] += dy
        verts[idx] = _hex_vertices(inner[idx][0], inner[idx][1], 1.0, inner[idx][2])
        
        if _all_disjoint(verts):
            ns = _min_outer_side(verts, cx, cy)
            if ns < best_side:
                best_side = ns
                best_inner = [list(x) for x in inner]
                step = max(0.003, step * 0.9998)
            else:
                inner[idx][0], inner[idx][1] = ox, oy
                verts[idx] = _hex_vertices(ox, oy, 1.0, inner[idx][2])
                step = min(0.2, step * 1.0005)
        else:
            inner[idx][0], inner[idx][1] = ox, oy
            verts[idx] = _hex_vertices(ox, oy, 1.0, inner[idx][2])
            step = min(0.2, step * 1.001)
    
    return best_inner, (cx, cy), best_side, 0.0
