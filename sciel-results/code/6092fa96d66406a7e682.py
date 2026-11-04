"""Packing 12 unit regular hexagons inside a minimal regular hexagon."""
import math
import time
import numpy as np

N = 12

# ---- geometry helpers (module level, cheap) ----
_HEX_DIRS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
_HEX_VERT = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(30, 360, 60)])
_TAU = 2.0 * math.pi


def _side_from_points(P):
    # P: (M,2). Returns (S, cx, cy) for the minimal enclosing regular hexagon (side-aligned).
    proj = P @ _HEX_DIRS.T            # (M,6)
    mx = proj.max(axis=0)
    mn = proj.min(axis=0)
    S = (mx + mn) / 3.0
    cx = (mx - mn) / 3.0
    cy = (mx - mn) / 3.0
    return S, cx, cy


def _objective(z):
    # z: (N,3) = [x, y, theta]
    S, cx, cy = _side_from_points(z[:, :2])
    # relative vertices in outer-hexagon frame
    R = z[:, 2][:, None]
    C = np.cos(R)
    Sn = np.sin(R)
    rot = np.empty((N, 2, 2))
    rot[:, 0, 0] = C
    rot[:, 0, 1] = -Sn
    rot[:, 1, 0] = Sn
    rot[:, 1, 1] = C
    # (N,2,6) vertices
    V = np.einsum('nij,njk->nik', rot, _HEX_VERT) + z[:, :2][:, None, :]
    V = V - np.array([cx, cy])
    # per-vertex, per-side signed distance (positive = inside)
    D = V @ _HEX_DIRS.T
    # margin = S - max over vertices of (|D| in each of the 6 directions)
    per = np.maximum(D, -D)           # (N,2,6)
    per_max = per.max(axis=1)         # (N,6)
    margin = S - per_max.max(axis=0)  # (6,)
    # penalty for negative margins (violation) with a smooth-ish square term
    viol = np.maximum(0.0, -margin)
    return float((viol * viol).sum())


def _feasible(z):
    S, cx, cy = _side_from_points(z[:, :2])
    R = z[:, 2][:, None]
    rot = np.empty((N, 2, 2))
    rot[:, 0, 0] = np.cos(R); rot[:, 0, 1] = -np.sin(R)
    rot[:, 1, 0] = np.sin(R); rot[:, 1, 1] = np.cos(R)
    V = np.einsum('nij,njk->nik', rot, _HEX_VERT) + z[:, :2][:, None, :]
    V = V - np.array([cx, cy])
    D = V @ _HEX_DIRS.T
    per = np.maximum(D, -D)
    per_max = per.max(axis=1)
    margin = S - per_max.max(axis=0)
    return bool((margin >= -1e-9).all()), S


def _refine(z0, budget):
    try:
        from scipy.optimize import minimize
        t0 = time.time()
        best = z0.copy()
        bestS = _side_from_points(z0[:, :2])[0]
        ok0, _ = _feasible(z0)
        if ok0:
            bestS = _side_from_points(z0[:, :2])[0]
        res = minimize(_objective, z0, method='L-BFGS-B',
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
        if res.x is not None:
            ok, S = _feasible(res.x)
            if ok and S < bestS:
                bestS = S; best = res.x
        # quick polish from best
        if time.time() - t0 < budget:
            res2 = minimize(_objective, best, method='L-BFGS-B',
                            options={'maxiter': 2000, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 50})
            if res2.x is not None:
                ok, S = _feasible(res2.x)
                if ok and S < bestS:
                    bestS = S; best = res2.x
        return best
    except Exception:
        return z0


def _initial_layouts():
    layouts = []
    # 1) straight line along x
    x0 = np.linspace(-4.0, 4.0, N)
    layouts.append(np.column_stack([x0, np.zeros(N), np.zeros(N)]))
    # 2) 3x4 grid
    xs = np.array([-1.5, -0.5, 0.5, 1.5])
    ys = np.array([-1.5, -0.5, 0.5, 1.5])
    pts = []
    for y in ys:
        for x in xs:
            pts.append((x, y, 0.0))
    layouts.append(np.array(pts))
    # 3) 4x3 grid
    xs = np.array([-1.5, -0.5, 0.5, 1.5])
    ys = np.array([-1.0, 0.0, 1.0])
    pts = []
    for y in ys:
        for x in xs:
            pts.append((x, y, 0.0))
    layouts.append(np.array(pts))
    # 4) 2x6 grid
    xs = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    ys = np.array([-1.0, 1.0])
    pts = []
    for y in ys:
        for x in xs:
            pts.append((x, y, 0.0))
    layouts.append(np.array(pts))
    # 5) ring + center-ish
    r = 1.8
    ang = np.linspace(0, _TAU, 12, endpoint=False)
    pts = [(r * math.cos(a), r * math.sin(a), 0.0) for a in ang]
    layouts.append(np.array(pts))
    # 6) two rows of 6, offset
    xs = np.linspace(-2.5, 2.5, 6)
    pts = []
    for x in xs:
        pts.append((x, -0.9, 0.0))
    for x in xs:
        pts.append((x + 0.5, 0.9, 0.0))
    layouts.append(np.array(pts))
    # 7) hexagonal packing: center + 6 + 5
    pts = [(0.0, 0.0, 0.0)]
    r1 = 1.7320508075688772
    for k in range(6):
        a = math.radians(60 * k)
        pts.append((r1 * math.cos(a), r1 * math.sin(a), 0.0))
    r2 = 3.0
    for k in range(5):
        a = math.radians(60 * k + 30)
        pts.append((r2 * math.cos(a), r2 * math.sin(a), 0.0))
    layouts.append(np.array(pts))
    return layouts


def construct():
    deadline = 120.0
    t0 = time.time()
    layouts = _initial_layouts()
    best = None
    bestS = float('inf')
    for z0 in layouts:
        if time.time() - t0 > deadline - 15.0:
            break
        z = _refine(z0, max(5.0, deadline - (time.time() - t0) - 20.0))
        ok, S = _feasible(z)
        if ok and S < bestS:
            bestS = S
            best = z
    if best is None:
        # deterministic fallback: line layout with generous side
        x0 = np.linspace(-4.0, 4.0, N)
        z = np.column_stack([x0, np.zeros(N), np.zeros(N)])
        S, cx, cy = _side_from_points(z[:, :2])
        S = max(S, 12.0)
        inner = [[float(x), float(y), float(a * 180.0 / math.pi)] for (x, y, a) in z]
        return inner, (0.0, 0.0), float(S), 0.0
    S, cx, cy = _side_from_points(best[:, :2])
    inner = [[float(x), float(y), float(a * 180.0 / math.pi)] for (x, y, a) in best]
    return inner, (float(cx), float(cy)), float(S), 0.0
