import time
import numpy as np
from scipy.optimize import differential_evolution

N = 12
_DEADLINE = time.time() + 125.0
_EPS = 1e-9
_ROT = np.arange(6) * np.pi / 3.0
_R3 = np.sqrt(3.0)


def _verts(x, y, ang):
    c = np.cos(ang)
    s = np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return R @ np.column_stack((np.cos(_ROT), np.sin(_ROT))) + np.array([x, y])


def _separated(A, B):
    for i in range(6):
        ai = A[i]
        aj = A[(i + 1) % 6]
        nrm = np.array([-(aj[1] - ai[1]), aj[0] - ai[0]])
        nn = np.linalg.norm(nrm)
        if nn < 1e-12:
            continue
        nrm /= nn
        d = nrm @ (B - ai)
        if np.max(d) <= _EPS:
            return True
    for j in range(6):
        bj = B[j]
        bk = B[(j + 1) % 6]
        nrm = np.array([-(bk[1] - bj[1]), bk[0] - bj[0]])
        nn = np.linalg.norm(nrm)
        if nn < 1e-12:
            continue
        nrm /= nn
        d = nrm @ (A - bj)
        if np.max(d) <= _EPS:
            return True
    return False


def _overlap(A, B):
    return not _separated(A, B)


def _in_hex(p, cx, cy, side, ang):
    c = np.cos(ang)
    s = np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    q = R @ (np.array(p) - np.array([cx, cy]))
    for i in range(6):
        pi_ = q
        pj = R @ np.array((np.cos(_ROT[(i + 1) % 6]), np.sin(_ROT[(i + 1) % 6])))
        nrm = np.array([-(pj[1] - pi_[1]), pj[0] - pi_[0]])
        nn = np.linalg.norm(nrm)
        if nn < 1e-12:
            continue
        nrm /= nn
        if nrm @ (q - pi_) > _EPS:
            return False
    return True


def _feasible(verts_list, cx, cy, side, ang):
    for i in range(len(verts_list)):
        if not all(_in_hex(v, cx, cy, side, ang) for v in verts_list[i]):
            return False
        for j in range(i + 1, len(verts_list)):
            if _overlap(verts_list[i], verts_list[j]):
                return False
    return True


def _min_side(verts_list, cx, cy, ang):
    lo = 0.0
    hi = 1.0
    for _ in range(34):
        mid = 0.5 * (lo + hi)
        if _feasible(verts_list, cx, cy, mid, ang):
            hi = mid
        else:
            lo = mid
    return hi


def _objective(z, cx, cy, ang):
    M = len(z) // 3
    verts = [_verts(z[3 * i], z[3 * i + 1], z[3 * i + 2]) for i in range(M)]
    if not _feasible(verts, cx, cy, 20.0, ang):
        return 1e6
    return _min_side(verts, cx, cy, ang)


def construct():
    global _DEADLINE
    t0 = time.time()

    # ---- Stage 1: place inner hexagons (fixed outer frame) ----
    cx, cy, ang = 0.0, 0.0, 0.0
    bounds = [(-14.0, 14.0)] * (3 * N)
    best = None
    best_val = np.inf
    while time.time() < _DEADLINE - 20.0:
        res = differential_evolution(
            _objective, bounds, args=(cx, cy, ang),
            seed=1234, maxiter=60, popsize=12, tol=1e-5,
            polish=False, init='sobol', updating='immediate',
            workers=1,
        )
        val = float(res.fun)
        if val < best_val:
            best_val = val
            best = res.x
        if best_val < 5.5:
            break
    if best is None:
        best = np.concatenate([[1.5 * (i - (N - 1) / 2.0), 0.0, 0.0] for i in range(N)])

    # ---- Stage 2: refine outer hexagon (center, side, rotation) ----
    verts = [_verts(best[3 * i], best[3 * i + 1], best[3 * i + 2]) for i in range(N)]
    o_bounds = [(-4.0, 4.0), (-4.0, 4.0), (3.0, 12.0), (0.0, np.pi / 3.0)]

    def _oobj(o):
        ocx, ocy, oside, oang = o
        if not _feasible(verts, ocx, ocy, oside, oang):
            return 1e6
        return oside

    best_o = np.array([0.0, 0.0, 10.0, 0.0])
    best_o_val = _oobj(best_o)
    while time.time() < _DEADLINE - 4.0:
        res = differential_evolution(
            _oobj, o_bounds, seed=42, maxiter=40, popsize=12,
            tol=1e-6, polish=True, init='sobol', updating='immediate', workers=1,
        )
        val = float(res.fun)
        if val < best_o_val:
            best_o_val = val
            best_o = res.x
        if best_o_val < 4.2:
            break

    # ---- Final polish of outer side via bisection ----
    ocx, ocy, oang = float(best_o[0]), float(best_o[1]), float(best_o[3])
    lo, hi = 0.0, float(best_o[2])
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if _feasible(verts, ocx, ocy, mid, oang):
            hi = mid
        else:
            lo = mid
    side = hi

    inner = [[float(best[3 * i]), float(best[3 * i + 1]), float(np.degrees(best[3 * i + 2]))]
             for i in range(N)]
    return inner, (ocx, ocy), float(side), float(oang)
