import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600.0

    def _remaining():
        return deadline - time.time()

    # ---- Fixed geometry of a unit regular hexagon ----
    # 3 edge normals (perpendicular to edges) at 30, 90, 150 deg
    na = np.deg2rad([30.0, 90.0, 150.0])
    normals = np.stack([np.cos(na), np.sin(na)], axis=1)          # (3,2)
    # 6 vertices at 0,60,...,300 deg
    va = np.deg2rad(np.arange(6) * 60.0)
    verts = np.stack([np.cos(va), np.sin(va)], axis=1)            # (6,2)
    # projection of each vertex on each normal  -> (3,6)
    VNP = normals @ verts.T
    # half-width (inradius) of a unit hexagon along each normal
    inr = np.max(np.abs(VNP), axis=1)                             # (3,)  == sqrt(3)/2

    # ---- smooth max via log-sum-exp ----
    def smax(A, s):
        m = A.max(axis=-1, keepdims=True)
        return (m + (1.0 / s) * np.log(np.sum(np.exp(s * (A - m)), axis=-1, keepdims=True))).squeeze(-1)

    # ---- objective: smooth min enclosing side + smooth overlap penalty ----
    def objective(x):
        P = x[:22].reshape(11, 2)
        C = x[22:24]
        R = P - C
        # enclosing side
        proj = R @ normals.T                                      # (11,3)
        S = proj + VNP                                            # (11,3,6)
        S = np.abs(S)
        M = smax(S, 250.0)                                        # (11,3)
        m = smax(M, 250.0)                                        # (11,)
        side = (2.0 / np.sqrt(3.0)) * smax(m, 250.0)
        # overlap penalty (smooth)
        pen = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                d = np.abs((P[i] - P[j]) @ normals.T)             # (3,)
                sep = d - inr                                     # must be > 0
                pen += smax(-sep, 300.0) ** 2
        return side + 100.0 * pen

    # ---- validity check (matches task contract exactly) ----
    def valid(P, C, side):
        if side < np.sqrt(11.0) - 1e-9:
            return False
        R = P - C
        # non-overlap: on at least one of 6 normals projected intervals apart by > 1e-6
        for i in range(11):
            for j in range(i + 1, 11):
                d = np.abs((P[i] - P[j]) @ normals.T)
                if d.max() < inr.max() + 1e-6:
                    return False
        # containment: every inner vertex inside outer hexagon
        Ov = side * verts
        for i in range(6):
            p1 = C + Ov[i]
            p2 = C + Ov[(i + 1) % 6]
            e = p2 - p1
            for h in range(11):
                for v in range(6):
                    p = P[h] + verts[v]
                    if np.cross(e, p - p1) < -1e-6:
                        return False
        # edge lengths within 1e-8 of 1 (unit hexagons -> always true)
        return True

    # ---- build a symmetric initial configuration ----
    def build(r1, r2, a1, a2, c):
        P = np.empty((11, 2))
        P[0] = c
        for k in range(6):
            ang = a1 + np.deg2rad(60.0 * k)
            P[1 + k] = c + r1 * np.array([np.cos(ang), np.sin(ang)])
        for k in range(4):
            ang = a2 + np.deg2rad(90.0 * k)
            P[7 + k] = c + r2 * np.array([np.cos(ang), np.sin(ang)])
        return P

    def side_of(P, C):
        R = P - C
        M = np.max(np.abs(R @ normals.T + VNP), axis=(0, 2))
        return (2.0 / np.sqrt(3.0)) * M.max()

    # ---- gather candidate starts ----
    starts = []
    for r1 in (2.06, 2.08, 2.10):
        for r2 in (3.95, 4.05, 4.15):
            for a1 in (0.0, 30.0):
                for a2 in (0.0, 45.0):
                    for c in (0.0, 0.25, -0.25):
                        P = build(r1, r2, a1, a2, np.array([c, 0.0]))
                        C = np.array([c, 0.0])
                        if valid(P, C, side_of(P, C)):
                            starts.append((P, C))
    if not starts:
        P = build(2.1, 4.2, 0.0, 0.0, np.array([0.0, 0.0]))
        C = np.array([0.0, 0.0])
        starts.append((P, C))

    # ---- optimize ----
    best = None
    for idx, (P, C) in enumerate(starts):
        if _remaining() < 35.0:
            break
        x0 = np.concatenate([P.ravel(), C.ravel()])
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 4000, 'xatol': 1e-10, 'fatol': 1e-10})
        except Exception:
            continue
        Pf = res.x[:22].reshape(11, 2)
        Cf = res.x[22:24]
        side = side_of(Pf, Cf)
        if valid(Pf, Cf, side):
            if best is None or side < best[2]:
                best = (Pf, Cf, side)

    # ---- high-precision polish on the best ----
    if best is not None and _remaining() > 40.0:
        P0, C0, _ = best
        x0 = np.concatenate([P0.ravel(), C0.ravel()])
        for tol in (1e-12, 1e-14):
            if _remaining() < 35.0:
                break
            try:
                res = minimize(objective, x0, method='Powell',
                               options={'maxiter': 3000, 'xatol': tol, 'fatol': tol})
            except Exception:
                break
            Pf = res.x[:22].reshape(11, 2)
            Cf = res.x[22:24]
            side = side_of(Pf, Cf)
            if valid(Pf, Cf, side) and side < best[2]:
                best = (Pf, Cf, side)
                x0 = res.x.copy()

    if best is None:
        P, C = starts[0]
        best = (P, C, side_of(P, C))

    P, C, side = best
    if side < np.sqrt(11.0):
        side = np.sqrt(11.0)

    inner = np.zeros((11, 3))
    inner[:, 0] = P[:, 0]
    inner[:, 1] = P[:, 1]
    inner[:, 2] = 0.0
    outer = np.array([C[0], C[1], 0.0])
    return inner, outer, float(side)
