import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 6.0
EPS = 1e-9

# Unit hexagon (flat-top) vertices
HEX_VERTS = np.array([
    [math.cos(math.radians(a)), math.sin(math.radians(a))]
    for a in range(0, 360, 60)
])

# 6 inward normals of the enclosing hexagon for rotation beta
def normals(beta):
    return np.array([
        [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
         math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
        for k in range(6)
    ])

def enclosing_D(pts, beta):
    """Max over vertices and over 6 normals of |proj| (half-width of enclosing hexagon)."""
    ns = normals(beta)
    best = 0.0
    for i in range(N):
        for j in range(6):
            p = pts[i] + HEX_VERTS[j]
            d = np.max(np.abs(p @ ns.T))
            if d > best:
                best = d
    return best

def enclosing_S(pts, beta):
    return (enclosing_D(pts, beta) + APOTHEM) / APOTHEM

def pair_min_gap(pts):
    """Min over pairs of (min over 6 normals of |d.n| - sqrt(3)). Positive => separated."""
    ns = normals(0.0)  # relative orientation fixed at 0 for identical-orientation hexes
    min_gap = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            d = pts[i] - pts[j]
            g = np.min(np.abs(d @ ns.T)) - math.sqrt(3)
            if g < min_gap:
                min_gap = g
    return min_gap

def feasible(pts):
    return pair_min_gap(pts) > -1e-9

def repulsion(pts, beta):
    """Total force on each hexagon from repulsion + soft containment."""
    ns = normals(beta)
    D = enclosing_D(pts, beta)
    forces = np.zeros((N, 2))
    for i in range(N):
        for j in range(i + 1, N):
            d = pts[i] - pts[j]
            # find the most-violated normal
            projs = np.abs(d @ ns.T)
            k = int(np.argmin(projs))
            n = ns[k]
            gap = projs[k] - math.sqrt(3)
            if gap < 0:
                f = -100.0 * gap  # push apart along n
                forces[i] += f * n
                forces[j] -= f * n
    # containment: pull inward if a vertex is beyond current D
    for i in range(N):
        for j in range(6):
            p = pts[i] + HEX_VERTS[j]
            projs = p @ ns.T
            k = int(np.argmax(np.abs(projs)))
            if abs(projs[k]) > D:
                n = ns[k]
                f = 20.0 * (abs(projs[k]) - D)
                forces[i] -= f * n
    return forces

def relax(pts, beta, max_iter=4000):
    """Simulated-annealing-like repulsion relaxation."""
    pts = pts.copy()
    T = 0.5
    step = 0.1
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        F = repulsion(pts, beta)
        # move
        pts_new = pts + step * F
        # anneal
        step *= 0.995
        T *= 0.999
        # accept if S improves or small
        if enclosing_S(pts_new, beta) <= enclosing_S(pts, beta) + 1e-6:
            pts = pts_new
        if step < 1e-6:
            break
    return pts

def optimize_beta(pts):
    """Find best beta (outer rotation) via coarse + fine scan + local refine."""
    best = (float('inf'), 0.0)
    # coarse
    for i in range(72):
        b = i * (math.pi / 6) / 72
        S = enclosing_S(pts, b)
        if S < best[0]:
            best = (S, b)
    # fine around best
    b0 = best[1]
    for i in range(121):
        b = b0 - 0.01 + i * 0.02 / 120
        S = enclosing_S(pts, b)
        if S < best[0]:
            best = (S, b)
    return best[1], best[0]

def center_hex(pts, beta):
    """Translate so enclosing hexagon is centered at origin."""
    ns = normals(beta)
    # compute center via LP: minimize D subject to -D <= p.n <= D
    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    for nx, ny in ns:
        for i in range(N):
            for j in range(6):
                px = pts[i][0] + HEX_VERTS[j][0]
                py = pts[i][1] + HEX_VERTS[j][1]
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0, None)]
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            return res.x[0], res.x[1], res.x[2]
    except Exception:
        pass
    return 0.0, 0.0, float('inf')

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Initial cluster: hexagonal lattice arrangement of 11 hexes (center + ring + 4)
    # Use axial coords converted to cartesian
    def axial(q, r):
        return (math.sqrt(3) * (q + r / 2.0), 1.5 * r)

    base_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2),
    ]
    base = np.array([axial(q, r) for q, r in base_axial])
    # center base
    base = base - base.mean(axis=0)

    best_S = float('inf')
    best_pts = None
    best_beta = 0.0
    best_center = (0.0, 0.0)

    # Multi-start: different initial perturbations
    rng = np.random.default_rng(42)
    starts = [base.copy()]
    for s in range(12):
        pert = base + rng.normal(0, 0.15, base.shape)
        starts.append(pert)

    for start in starts:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        # relax with a few beta guesses
        for beta0 in [0.0, math.pi / 12, math.pi / 6]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            pts = relax(start, beta0)
            beta, S = optimize_beta(pts)
            if S < best_S:
                best_S = S
                best_pts = pts.copy()
                best_beta = beta

    # Final polish: re-relax from best, re-optimize beta, center
    if best_pts is not None:
        pts = relax(best_pts, best_beta)
        beta, S = optimize_beta(pts)
        if S < best_S:
            best_S = S
            best_pts = pts
            best_beta = beta
        cx, cy, D = center_hex(best_pts, best_beta)
        best_center = (cx, cy)
        best_S = (D + APOTHEM) / APOTHEM

    if best_pts is None:
        best_pts = base
        best_beta = 0.0
        cx, cy, D = center_hex(best_pts, best_beta)
        best_center = (cx, cy)
        best_S = (D + APOTHEM) / APOTHEM

    # Shift points by center so enclosing hexagon centered at origin
    pts_final = best_pts - np.array(best_center)

    angle_deg = math.degrees(best_beta)
    inner = []
    for x, y in pts_final:
        inner.append([float(x), float(y), float(angle_deg)])
    center = [float(best_center[0]), float(best_center[1])]
    side = float(best_S)
    rotation = float(angle_deg)
    return inner, center, side, rotation
