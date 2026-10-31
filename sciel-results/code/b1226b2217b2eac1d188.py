import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hex_vertices(cx, cy, angle_deg, side=1.0):
    a = np.radians(angle_deg)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    r = side
    vx = r * np.cos(base_angles + a)
    vy = r * np.sin(base_angles + a)
    return np.column_stack([cx + vx, cy + vy])

def hexagon_containment_s(px, py):
    sqrt3 = np.sqrt(3)
    vals = np.array([
        (sqrt3/2)*px + (1/2)*py,
        py,
        -(sqrt3/2)*px + (1/2)*py,
        -(sqrt3/2)*px - (1/2)*py,
        -py,
        (sqrt3/2)*px - (1/2)*py
    ])
    return (2.0/sqrt3) * np.max(vals)

def sat_penetration(A, B):
    min_overlap = np.inf
    for poly in [A, B]:
        for i in range(6):
            v1 = poly[i]
            v2 = poly[(i+1) % 6]
            edge = v2 - v1
            n = np.array([-edge[1], edge[0]])
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn
            proj_A = A @ n
            proj_B = B @ n
            max_A, min_A = proj_A.max(), proj_A.min()
            max_B, min_B = proj_B.max(), proj_B.min()
            if max_A < min_B - 1e-10 or max_B < min_A - 1e-10:
                return 0.0
            overlap = min(max_A, max_B) - max(min_A, min_B)
            if overlap < min_overlap:
                min_overlap = overlap
    if min_overlap == np.inf:
        return 0.0
    return max(0.0, min_overlap)

def compute_outer_side(params):
    max_s = 0.0
    for i in range(N):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        v = hex_vertices(x, y, a)
        for j in range(6):
            s = hexagon_containment_s(v[j,0], v[j,1])
            if s > max_s:
                max_s = s
    return max_s

def objective(params):
    penalty_w = 100.0
    verts = []
    for i in range(N):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        verts.append(hex_vertices(x, y, a))
    
    max_s = 0.0
    for v in verts:
        for j in range(6):
            s = hexagon_containment_s(v[j,0], v[j,1])
            if s > max_s:
                max_s = s
    
    pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = sat_penetration(verts[i], verts[j])
            if d > 0:
                pen += d * d
    
    return max_s + penalty_w * pen

def _extract(params):
    inner = []
    max_s = 0.0
    for i in range(N):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        inner.append([float(x), float(y), float(a)])
        v = hex_vertices(x, y, a)
        for j in range(6):
            s = hexagon_containment_s(v[j,0], v[j,1])
            if s > max_s:
                max_s = s
    return inner, (0.0, 0.0), float(max_s), 0.0

def construct():
    t0 = time.time()
    deadline = t0 + 135.0
    
    sqrt3 = np.sqrt(3)
    spacing = sqrt3
    
    # Triangular lattice honeycomb init: 3-4-3-1
    init = []
    y0 = 0.0
    for i in range(3):
        init.append([(i-1)*spacing, y0, 0.0])
    y1 = spacing * sqrt3 / 2
    for i in range(4):
        init.append([(i-1.5)*spacing, y1, 0.0])
    y2 = 2 * y1
    for i in range(3):
        init.append([(i-1)*spacing, y2, 0.0])
    y3 = 3 * y1
    init.append([0.0, y3, 0.0])
    
    p0 = np.array(init).flatten()
    best_x = p0.copy()
    best_f = objective(p0)
    
    # Multiple starts
    np.random.seed(42)
    starts = [p0]
    for _ in range(10):
        starts.append(p0 + np.random.randn(3*N) * 0.5)
    
    for p_init in starts:
        if time.time() > deadline - 8:
            break
        try:
            res = minimize(objective, p_init, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            f = res.fun
            if f < best_f:
                best_f = f
                best_x = res.x
        except Exception:
            pass
    
    return _extract(best_x)
