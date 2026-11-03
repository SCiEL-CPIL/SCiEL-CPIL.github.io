import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_DEADLINE = 150.0
_T0 = time.time()

# ---------- geometry ----------
def _hex_verts(cx, cy, s, ang):
    a = np.deg2rad(ang)
    c, sn = math.cos(a), math.sin(a)
    pts = []
    for i in range(6):
        t = math.pi / 3.0 * i + a
        x = s * math.cos(t)
        y = s * math.sin(t)
        pts.append((cx + c * x - sn * y, cy + sn * x + c * y))
    return np.array(pts)


def _sep(p, q, e):
    # minimum distance between two convex polygons (Minkowski sum)
    d = p - q
    n1, n2 = len(p), len(q)
    best = math.inf
    for i in range(n1):
        v1 = p[(i + 1) % n1] - p[i]
        L1 = np.hypot(v1[0], v1[1])
        n1u = np.array([-v1[1], v1[0]]) / L1
        for j in range(n2):
            v2 = q[(j + 1) % n2] - q[j]
            L2 = np.hypot(v2[0], v2[1])
            n2u = np.array([-v2[1], v2[0]]) / L2
            n = n1u + n2u
            Ln = np.hypot(n[0], n[1])
            if Ln < 1e-12:
                continue
            n = n / Ln
            c = d @ n
            if c <= 0:
                best = 0.0
                return best
            t1 = (d @ n1u) / L1
            t2 = (d @ n2u) / L2
            if 0 <= t1 <= 1 and 0 <= t2 <= 1:
                best = 0.0
                return best
            if c < best:
                best = c
    return best


def _pair_dist(A, B):
    a = _hex_verts(A[0], A[1], 1.0, A[2])
    b = _hex_verts(B[0], B[1], 1.0, B[2])
    return _sep(a, b)


def _pair_barrier(A, B, gamma=1.0):
    d = _pair_dist(A, B)
    if d >= gamma:
        return 0.0
    return (gamma - d) ** 2


# ---------- objective: min enclosing side + non-overlap barrier ----------
def _objective(x, gamma, it):
    n = N
    pos = x[0:2 * n].reshape(n, 2)
    ang = x[2 * n:3 * n]
    # barrier
    bar = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            bar += _pair_barrier((pos[i][0], pos[i][1], ang[i]),
                                 (pos[j][0], pos[j][1], ang[j]), gamma)
    # LP for min enclosing regular hexagon (center, side, rotation free)
    A_ub, b_ub = [], []
    for i in range(n):
        px, py, th = pos[i][0], pos[i][1], ang[i]
        for k in range(6):
            t = math.pi / 3.0 * k
            ct, st = math.cos(t), math.sin(t)
            # corner = s*(ct,st) rotated by th
            rct = ct * math.cos(th) - st * math.sin(th)
            rst = ct * math.sin(th) + st * math.cos(th)
            # rct*cx + rst*cy + rct*s - st*s <= px  (corner.x - px <= 0)
            A_ub.append([rct, rst, rct - st, 0, 0])
            b_ub.append(px)
            # -rst*cx + rct*cy + rst*s - ct*s <= py
            A_ub.append([-rst, rct, rst - ct, 0, 0])
            b_ub.append(py)
    c = [0, 0, 1, 0, 0]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    if res.status != 0:
        return 1e6 + bar
    s = max(res.x[2], 0.0)
    return s + 1000.0 * bar


def construct():
    t0 = time.time()
    best = None
    best_score = math.inf
    best_state = None

    def _deadline():
        return (time.time() - t0) > _DEADLINE - 8.0

    def _try_state(pos, ang, gamma):
        nonlocal best, best_score, best_state
        x = np.concatenate([pos.reshape(-1), ang])
        obj = _objective(x, gamma, 0)
        s = obj - 1000.0 * (obj - obj)  # placeholder
        # recompute s cleanly
        # (we already have s inside; approximate by objective minus barrier)
        # recompute barrier to isolate s
        bar = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                bar += _pair_barrier((pos[i][0], pos[i][1], ang[i]),
                                     (pos[j][0], pos[j][1], ang[j]), gamma)
        s = obj - 1000.0 * bar
        if s < best_score:
            best_score = s
            best_state = (pos.copy(), ang.copy())
            best = (best_state[0], best_state[1])
        return obj

    # ---- initial placements ----
    inits = []
    # line
    p = np.zeros((N, 2)); a = np.zeros(N)
    for k in range(N):
        p[k] = [4.0 * (k - N // 2), 0.0]
    inits.append((p, a))
    # hexagonal cluster
    idx = [(0, 0)]
    for r in range(1, 5):
        for aq in range(6):
            idx.append((r * math.cos(math.pi / 3 * aq), r * math.sin(math.pi / 3 * aq)))
    idx = idx[:N]
    p = np.array([[4.0 * ix, 4.0 * iy] for ix, iy in idx]); a = np.zeros(N)
    inits.append((p, a))
    # grid
    p = np.zeros((N, 2)); a = np.zeros(N)
    for k in range(N):
        p[k] = [4.0 * (k % 3), 4.0 * (k // 3)]
    inits.append((p, a))
    # ring
    p = np.zeros((N, 2)); a = np.zeros(N)
    for k in range(N):
        t = 2 * math.pi * k / N
        p[k] = [3.0 * math.cos(t), 3.0 * math.sin(t)]
    inits.append((p, a))

    for (p0, a0) in inits:
        if _deadline():
            break
        pos = p0.copy().astype(float)
        ang = a0.copy().astype(float)
        x = np.concatenate([pos.reshape(-1), ang])
        gamma = 0.5
        # outer loop: shrink barrier
        for gstep in range(8):
            if _deadline():
                break
            gamma = max(0.01, gamma * 0.5)
            res = minimize(_objective, x, args=(gamma, gstep),
                           method='L-BFGS-B',
                           options={'maxiter': 40, 'maxls': 10, 'gtol': 1e-4})
            x = res.x
            pos = x[0:2 * N].reshape(N, 2)
            ang = x[2 * N:3 * N]
            _try_state(pos, ang, gamma)

    # ---- final polish: exact LP on best state, then a short tight refine ----
    if best_state is not None and not _deadline():
        pos, ang = best_state
        x = np.concatenate([pos.reshape(-1), ang])
        for _ in range(3):
            if _deadline():
                break
            res = minimize(_objective, x, args=(0.0, 99),
                           method='L-BFGS-B',
                           options={'maxiter': 20, 'maxls': 5, 'gtol': 1e-6})
            x = res.x
            pos = x[0:2 * N].reshape(N, 2)
            ang = x[2 * N:3 * N]
            _try_state(pos, ang, 0.0)

    if best is None:
        # fallback: seed-like
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    pos, ang = best
    # final exact LP to get center & rotation for the score
    A_ub, b_ub = [], []
    for i in range(N):
        px, py, th = pos[i][0], pos[i][1], ang[i]
        for k in range(6):
            t = math.pi / 3.0 * k
            ct, st = math.cos(t), math.sin(t)
            rct = ct * math.cos(th) - st * math.sin(th)
            rst = ct * math.sin(th) + st * math.cos(th)
            A_ub.append([rct, rst, rct - st, 0, 0])
            b_ub.append(px)
            A_ub.append([-rst, rct, rst - ct, 0, 0])
            b_ub.append(py)
    res = linprog([0, 0, 1, 0, 0], A_ub=A_ub, b_ub=b_ub, method='highs')
    if res.status == 0:
        cx, cy, s = float(res.x[0]), float(res.x[1]), max(float(res.x[2]), 0.0)
    else:
        cx, cy, s = 0.0, 0.0, 40.0
    inner = [[float(pos[i][0]), float(pos[i][1]), float(ang[i])] for i in range(N)]
    return inner, (cx, cy), s, 0.0
