import numpy as np
from scipy.optimize import linprog
import random, math

N = 11

def hex_vertices(r, cx, cy, angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    v = []
    for k in range(6):
        th = math.radians(60.0 * k)
        px, py = r * math.cos(th), r * math.sin(th)
        v.append((cx + c * px - s * py, cy + s * px + c * py))
    return v

def hex_points(r, cx, cy, angle_deg, m=16):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    pts = []
    for k in range(6 * m):
        th = 2.0 * math.pi * k / (6 * m)
        px, py = r * math.cos(th), r * math.sin(th)
        pts.append((cx + c * px - s * py, cy + s * px + c * py))
    return pts

def hexagon_corners(r, cx, cy, angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    out = []
    for k in range(6):
        th = math.radians(60.0 * k)
        px, py = r * math.cos(th), r * math.sin(th)
        out.append((cx + c * px - s * py, cy + s * px + c * py))
    return out

def min_hex_side(pts, angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    P = np.array(pts)
    u = np.array([c, s])
    w = np.array([-s, c])
    t1 = P @ u; t2 = P @ w
    d1 = t1.max() - t1.min(); d2 = t2.max() - t2.min()
    return 0.5 * (d1 + d2) / math.sqrt(3.0)

def cluster_extents(r, centers, angles):
    corners = []
    for (cx, cy), ang in zip(centers, angles):
        corners.extend(hexagon_corners(r, cx, cy, ang))
    return corners

def min_side_for_orientation(r, centers, angles, angle_deg):
    corners = cluster_extents(r, centers, angles)
    return min_hex_side(corners, angle_deg)

def min_side_over_orientations(r, centers, angles, n=36):
    corners = cluster_extents(r, centers, angles)
    best = float('inf')
    for i in range(n):
        ang = 360.0 * i / n
        best = min(best, min_hex_side(corners, ang))
    return best

def min_enclosing_lp(corners, angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    u = np.array([c, s]); w = np.array([-s, c])
    P = np.array(corners)
    t1 = P @ u; t2 = P @ w
    m1, M1 = t1.min(), t1.max()
    m2, t2min, M2 = t2.min(), t2.min(), t2.max()
    A = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    b = np.array([M1 + M2, -M1 + M2, M1 - M2, -M1 - M2])
    res = linprog(np.zeros(2), A_ub=A, b_ub=b, bounds=[(0, None), (0, None)], method='highs')
    if res.success:
        return 0.5 * (res.x[0] + res.x[1]) / math.sqrt(3.0)
    return 0.5 * ((M1 - m1) + (M2 - m2)) / math.sqrt(3.0)

def min_side_lp(r, centers, angles, n=36):
    corners = cluster_extents(r, centers, angles)
    best = float('inf')
    for i in range(n):
        ang = 360.0 * i / n
        best = min(best, min_enclosing_lp(corners, ang))
    return best

def make_ring(r):
    centers = [(0.0, 0.0)]
    angles = [0.0]
    for k in range(6):
        th = math.radians(60.0 * k)
        d = 2.0 * math.sqrt(3.0) * r
        centers.append((d * math.cos(th), d * math.sin(th)))
        angles.append(0.0)
    for k in range(4):
        th = math.radians(60.0 * (k + 1))
        d = 2.0 * math.sqrt(3.0) * r * math.sqrt(3.0)
        centers.append((d * math.cos(th), d * math.sin(th)))
        angles.append(0.0)
    return centers, angles

def ring_score(r, centers, angles):
    return min_side_over_orientations(r, centers, angles)

def optimize_ring(r):
    centers, angles = make_ring(r)
    best = (min_side_over_orientations(r, centers, angles), centers, angles)
    rng = random.Random(12345)
    for _ in range(250):
        c2 = [list(c) for c in centers]
        a2 = list(angles)
        for i in range(len(c2)):
            c2[i][0] += rng.uniform(-0.25, 0.25)
            c2[i][1] += rng.uniform(-0.25, 0.25)
            a2[i] = (a2[i] + rng.uniform(-20, 20)) % 360.0
        sc = min_side_over_orientations(r, c2, a2)
        if sc < best[0] - 1e-9:
            best = (sc, [tuple(c) for c in c2], a2)
    return best

def optimize_ring_lp(r):
    centers, angles = make_ring(r)
    best = (min_side_lp(r, centers, angles), centers, angles)
    rng = random.Random(777)
    for _ in range(150):
        c2 = [list(c) for c in centers]
        a2 = list(angles)
        for i in range(len(c2)):
            c2[i][0] += rng.uniform(-0.2, 0.2)
            c2[i][1] += rng.uniform(-0.2, 0.2)
            a2[i] = (a2[i] + rng.uniform(-15, 15)) % 360.0
        sc = min_side_lp(r, c2, a2)
        if sc < best[0] - 1e-9:
            best = (sc, [tuple(c) for c in c2], a2)
    return best

def construct():
    r = 1.0
    cands = []
    try:
        sc, centers, angles = optimize_ring(r)
        cands.append((sc, centers, angles))
    except Exception:
        pass
    try:
        sc, centers, angles = optimize_ring_lp(r)
        cands.append((sc, centers, angles))
    except Exception:
        pass
    if not cands:
        centers, angles = make_ring(r)
        cands.append((min_side_over_orientations(r, centers, angles), centers, angles))
    cands.sort(key=lambda x: x[0])
    sc, centers, angles = cands[0]
    inner = [[cx, cy, ang] for (cx, cy), ang in zip(centers, angles)]
    return inner, (0.0, 0.0), sc, 0.0
