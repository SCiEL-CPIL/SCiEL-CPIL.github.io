"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import linprog

N = 11

# EVOLVE-BLOCK-START

def _hex_verts(cx, cy, s, theta):
    a = theta + np.arange(6) * (np.pi / 3.0)
    return np.column_stack((cx + s * np.cos(a), cy + s * np.sin(a)))


def _hex_halfplanes(cx, cy, s, theta):
    apothem = s * math.sqrt(3) / 2.0
    A = np.zeros((6, 2))
    b = np.zeros(6)
    for i in range(6):
        ang = theta + (i + 0.5) * math.pi / 3.0
        nx = math.cos(ang)
        ny = math.sin(ang)
        A[i, 0] = nx
        A[i, 1] = ny
        b[i] = nx * cx + ny * cy + apothem
    return A, b


def _separation_distance(v1, v2, v3, v4):
    p = v1
    r = v2 - v1
    q = v3
    s = v4 - v3

    d = p - q
    a = np.dot(r, r)
    e = np.dot(s, s)
    f = np.dot(s, d)

    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(d))

    if a < 1e-12:
        u = float(np.clip(f / e, 0.0, 1.0))
        dist_sq = np.dot(d + u * s, d + u * s)
        return math.sqrt(max(0.0, dist_sq))

    c = np.dot(r, d)
    if e < 1e-12:
        t = float(np.clip(-c / a, 0.0, 1.0))
        dist_sq = np.dot(d - t * r, d - t * r)
        return math.sqrt(max(0.0, dist_sq))

    b = np.dot(r, s)
    denom = a * e - b * b
    if denom < 1e-12:
        t = float(np.clip(-c / a, 0.0, 1.0))
        dist_sq = np.dot(d - t * r, d - t * r)
        return math.sqrt(max(0.0, dist_sq))

    t = float(np.clip((b * f - c * e) / denom, 0.0, 1.0))
    u1 = (b * t + f) / e
    if u1 < 0.0:
        u1 = 0.0
        t = float(np.clip(-c / a, 0.0, 1.0))
    elif u1 > 1.0:
        u1 = 1.0
        t = float(np.clip((b - c) / a, 0.0, 1.0))

    dist_sq = np.dot(d - t * r + u1 * s, d - t * r + u1 * s)
    return math.sqrt(max(0.0, dist_sq))


def _check_disjoint(hex1, hex2):
    cx1, cy1, th1, s1 = hex1
    cx2, cy2, th2, s2 = hex2
    
    r1 = s1 * math.sqrt(3)
    r2 = s2 * math.sqrt(3)
    if abs(cx1 - cx2) > r1 + r2 or abs(cy1 - cy2) > r1 + r2:
        return True
    
    A1, b1 = _hex_halfplanes(cx1, cy1, s1, th1)
    A2, b2 = _hex_halfplanes(cx2, cy2, s2, th2)
    
    if np.all(A1 @ [cx2, cy2] <= b1 + 1e-9):
        return False
    if np.all(A2 @ [cx1, cy1] <= b2 + 1e-9):
        return False
    
    verts1 = _hex_verts(cx1, cy1, s1, th1)
    verts2 = _hex_verts(cx2, cy2, s2, th2)
    
    for i in range(6):
        for j in range(6):
            dist = _separation_distance(verts1[i], verts1[(i+1)%6], verts2[j], verts2[(j+1)%6])
            if dist < 1e-9:
                return False
    return True


def _find_min_outer_hex(inner_hexes):
    all_pts = []
    for (cx, cy, theta, s) in inner_hexes:
        verts = _hex_verts(cx, cy, s, theta)
        all_pts.extend(verts.tolist())
    all_pts = np.array(all_pts)
    
    best_side = float('inf')
    best_params = None
    
    for theta_outer in np.linspace(0, math.pi/3, 13):
        A = []
        b = []
        for i in range(6):
            ang = theta_outer + (i + 0.5) * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            for p in all_pts:
                A.append([-nx, -ny, -1.0])
                b.append(-nx * p[0] - ny * p[1])
        
        A = np.array(A)
        b = np.array(b)
        c = np.array([0.0, 0.0, 1.0])
        
        res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)], method='highs')
        
        if res.status == 0:
            apothem = res.x[2]
            side = apothem * 2.0 / math.sqrt(3)
            if side < best_side:
                best_side = side
                best_params = (res.x[0], res.x[1], side, theta_outer)
    
    return best_params, best_side


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    dx = math.sqrt(3)
    dy = 1.5
    
    configs = []
    
    cfg = [(k * dx, 0.0, 0.0, 1.0) for k in range(N)]
    configs.append(cfg)
    
    cfg = [(0.0, 0.0, 0.0, 1.0)]
    for i in range(6):
        ang = i * math.pi / 3.0
        cfg.append((dx * math.cos(ang), dx * math.sin(ang), 0.0, 1.0))
    for i in range(4):
        ang = i * math.pi / 3.0 + math.pi / 6.0
        cfg.append((2 * dx * math.cos(ang), 2 * dx * math.sin(ang), 0.0, 1.0))
    configs.append(cfg)
    
    cfg = []
    for j in range(3):
        cfg.append(((j - 1) * dx, 0.0, 0.0, 1.0))
    for j in range(4):
        cfg.append(((j - 1.5) * dx, dy, 0.0, 1.0))
    for j in range(3):
        cfg.append(((j - 1) * dx, 2 * dy, 0.0, 1.0))
    configs.append(cfg)
    
    cfg = []
    for j in range(2):
        cfg.append(((j - 0.5) * dx, 0.0, 0.0, 1.0))
    for j in range(3):
        cfg.append(((j - 1) * dx, dy, 0.0, 1.0))
    for j in range(3):
        cfg.append(((j - 1) * dx, 2 * dy, 0.0, 1.0))
    for j in range(3):
        cfg.append(((j - 1) * dx, 3 * dy, 0.0, 1.0))
    configs.append(cfg)
    
    cfg = []
    cfg.append((0.0, 0.0, 0.0, 1.0))
    for j in range(2):
        cfg.append(((j - 0.5) * dx, dy, 0.0, 1.0))
    for j in range(3):
        cfg.append(((j - 1) * dx, 2 * dy, 0.0, 1.0))
    for j in range(3):
        cfg.append(((j - 1) * dx, 3 * dy, 0.0, 1.0))
    for j in range(2):
        cfg.append(((j - 0.5) * dx, 4 * dy, 0.0, 1.0))
    configs.append(cfg)
    
    best_side = float('inf')
    best_inner = None
    best_outer = None
    
    for config in configs:
        if time.time() > deadline:
            break
        
        valid = True
        for i in range(len(config)):
            for j in range(i + 1, len(config)):
                if not _check_disjoint(config[i], config[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        params, side = _find_min_outer_hex(config)
        
        if params is not None and side < best_side:
            best_side = side
            best_inner = config
            best_outer = params
    
    if best_inner is None:
        config = configs[0]
        all_pts = []
        for (cx, cy, theta, s) in config:
            verts = _hex_verts(cx, cy, s, theta)
            all_pts.extend(verts.tolist())
        all_pts = np.array(all_pts)
        cx = (all_pts[:, 0].max() + all_pts[:, 0].min()) / 2
        cy = (all_pts[:, 1].max() + all_pts[:, 1].min()) / 2
        max_dist = np.max(np.linalg.norm(all_pts - [cx, cy], axis=1))
        side = max_dist * 2.0 / math.sqrt(3) * 1.2
        best_inner = config
        best_outer = (cx, cy, side, 0.0)
    
    cx, cy, side, theta = best_outer
    inner = [[h[0], h[1], math.degrees(h[2])] for h in best_inner]
    
    return inner, (cx, cy), side, math.degrees(theta)

# EVOLVE-BLOCK-END
