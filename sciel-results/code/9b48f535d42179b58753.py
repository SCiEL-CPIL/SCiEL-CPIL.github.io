import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 143.0
SQRT3 = math.sqrt(3)

# Unit hexagon vertices (flat-top, side=1): vertices at angles 0,60,...,300
_HEX_ANGS = np.radians(np.arange(0, 360, 60))
_UX = np.cos(_HEX_ANGS)
_UY = np.sin(_HEX_ANGS)

# Edge outward normals for unit hexagon
_ENX = np.zeros(6)
_ENY = np.zeros(6)
for _i in range(6):
    _ex = _UX[(_i+1) % 6] - _UX[_i]
    _ey = _UY[(_i+1) % 6] - _UY[_i]
    _nx, _ny = -_ey, _ex
    _n = math.sqrt(_nx * _nx + _ny * _ny)
    _ENX[_i] = _nx / _n
    _ENY[_i] = _ny / _n


def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)

    def verts_of(px, py, a_rad):
        c, s = math.cos(a_rad), math.sin(a_rad)
        vx = px + _UX * c - _UY * s
        vy = py + _UX * s + _UY * c
        return vx, vy

    def disjoint_fast(px1, py1, a1, px2, py2, a2):
        dx = px1 - px2
        dy = py1 - py2
        if dx * dx + dy * dy > 4.5:
            return True
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        v1x = px1 + _UX * c1 - _UY * s1
        v1y = py1 + _UX * s1 + _UY * c1
        v2x = px2 + _UX * c2 - _UY * s2
        v2y = py2 + _UX * s2 + _UY * c2
        e1x = _ENX * c1 - _ENY * s1
        e1y = _ENX * s1 + _ENY * c1
        e2x = _ENX * c2 - _ENY * s2
        e2y = _ENX * s2 + _ENY * c2
        for i in range(6):
            nx, ny = e1x[i], e1y[i]
            if (v1x * nx + v1y * ny).max() < (v2x * nx + v2y * ny).min() or \
               (v2x * nx + v2y * ny).max() < (v1x * nx + v1y * ny).min():
                return True
        for i in range(6):
            nx, ny = e2x[i], e2y[i]
            if (v1x * nx + v1y * ny).max() < (v2x * nx + v2y * ny).min() or \
               (v2x * nx + v2y * ny).max() < (v1x * nx + v1y * ny).min():
                return True
        return False

    def all_valid(pos, ang_rad):
        for i in range(N):
            for j in range(i + 1, N):
                if not disjoint_fast(pos[i, 0], pos[i, 1], ang_rad[i],
                                     pos[j, 0], pos[j, 1], ang_rad[j]):
                    return False
        return True

    def enclosing_side(pos, ang_rad):
        vx = np.zeros(N * 6)
        vy = np.zeros(N * 6)
        for i in range(N):
            c, s = math.cos(ang_rad[i]), math.sin(ang_rad[i])
            vx[i*6:(i+1)*6] = pos[i, 0] + _UX * c - _UY * s
            vy[i*6:(i+1)*6] = pos[i, 1] + _UX * s + _UY * c
        cx = vx.mean()
        cy = vy.mean()
        ox = vx - cx
        oy = vy - cy

        def phi_val(phi_deg):
            phi = math.radians(phi_deg)
            na = phi + np.pi / 6 + np.arange(6) * np.pi / 3
            nx = np.cos(na)
            ny = np.sin(na)
            proj = ox[:, None] * nx[None, :] + oy[:, None] * ny[None, :]
            return (2.0 / SQRT3) * np.max(np.abs(proj))

        best = float('inf')
        best_phi = 0.0
        for p in range(0, 60, 1):
            v = phi_val(p)
            if v < best:
                best = v
                best_phi = p
        try:
            lo = max(0.0, best_phi - 1.0)
            hi = min(60.0, best_phi + 1.0)
            r = minimize_scalar(phi_val, bounds=(lo, hi), method='bounded',
                                options={'xatol': 1e-10})
            if r.success and r.fun < best:
                best = r.fun
                best_phi = r.x
        except Exception:
            pass
        return best, best_phi, cx, cy

    # --- Initial configurations ---
    def init_honeycomb():
        p = np.zeros((N, 2))
        a = np.zeros(N)
        a1x, a1y = SQRT3, 0.0
        a2x, a2y = SQRT3 / 2, 1.5
        coords = [(0,0),(1,0),(-1,0),(0,1),(0,-1),
                  (1,-1),(-1,1),(2,-1),(-2,1),(1,1),(-1,-1),(2,0)]
        for i, (m, n) in enumerate(coords):
            p[i, 0] = m * a1x + n * a2x
            p[i, 1] = m * a1y + n * a2y
        return p, a

    def init_radial():
        p = np.zeros((N, 2))
        a = np.zeros(N)
        p[0] = [0, 0]
        for k in range(6):
            ang = k * np.pi / 3
            p[1+k] = [2 * SQRT3 * math.cos(ang), 2 * SQRT3 * math.sin(ang)]
        for k in range(5):
            ang = k * 2 * np.pi / 5 + np.pi / 5
            p[7+k] = [3 * SQRT3 * math.cos(ang), 3 * SQRT3 * math.sin(ang)]
        return p, a

    def init_rows():
        p = np.zeros((N, 2))
        a = np.zeros(N)
        for k in range(6):
            p[k] = [k * 2 * SQRT3, 0]
            p[k + 6] = [k * 2 * SQRT3 + SQRT3 / 2, 3.0]
        return p, a

    inits = [init_honeycomb(), init_radial(), init_rows()]

    best_side = float('inf')
    best_pos = None
    best_ang = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    for pi, ai in inits:
        if not all_valid(pi, np.radians(ai)):
            continue
        s, phi, cx, cy = enclosing_side(pi, np.radians(ai))
        if s < best_side:
            best_side = s
            best_pos = pi.copy()
            best_ang = ai.copy()
            best_phi = phi
            best_cx = cx
            best_cy = cy

    if best_pos is None:
        best_pos, best_ang = init_honeycomb()
        best_side, best_phi, best_cx, best_cy = enclosing_side(best_pos, np.radians(best_ang))

    # --- Phase 1: SA with adaptive step ---
    t_phase1 = time.time()
    sa_budget = min(DEADLINE * 0.50, DEADLINE - (t0))
    cur_pos = best_pos.copy()
    cur_ang = best_ang.copy()
    cur_side, cur_phi, cur_cx, cur_cy = enclosing_side(cur_pos, np.radians(cur_ang))

    T = 0.4
    T_min = 0.00002
    cool = 0.99996
    step = 0.30
    step_min = 0.002
    step_max = 0.8
    acc = 0
    tot = 0
    max_steps = 300000

    while time.time() - t0 < sa_budget:
        idx = random.randrange(N)
        np_ = cur_pos.copy()
        na_ = cur_ang.copy()
        np_[idx, 0] += random.gauss(0, step)
        np_[idx, 1] += random.gauss(0, step)
        na_[idx] += random.gauss(0, 5.0)

        ok = True
        ar = np.radians(na_)
        for j in range(N):
            if j == idx:
                continue
            if not disjoint_fast(np_[idx, 0], np_[idx, 1], ar[idx],
                                 np_[j, 0], np_[j, 1], ar[j]):
                ok = False
                break

        if not ok:
            T = max(T_min, T * cool)
            step = max(step_min, step * 0.99998)
            continue

        ns, np_, nphi, ncx, ncy = enclosing_side(np_, ar)
        if ns < cur_side:
            cur_pos, cur_ang = np_, na_
            cur_side, cur_phi, cur_cx, cur_cy = ns, nphi, ncx, ncy
            acc += 1
            if ns < best_side:
                best_side = ns
                best_pos = np_.copy()
                best_ang = na_.copy()
                best_phi, best_cx, best_cy = nphi, ncx, ncy
        else:
            d = max(0.0, ns - cur_side)
            if random.random() < math.exp(-d / T):
                cur_pos, cur_ang = np_, na_
                cur_side, cur_phi, cur_cx, cur_cy = ns, nphi, ncx, ncy
                acc += 1
                if ns < best_side:
                    best_side = ns
                    best_pos = np_.copy()
                    best_ang = na_.copy()
                    best_phi, best_cx, best_cy = nphi, ncx, ncy

        tot += 1
        if tot % 400 == 0:
            rate = acc / tot
            if rate < 0.12:
                step = max(step_min, step * 0.82)
            elif rate > 0.45:
                step = min(step_max, step * 1.18)
            acc = 0
            tot = 0
        T = max(T_min, T * cool)
        step = max(step_min, step * 0.99998)

    # --- Phase 2: Coordinate descent polish ---
    rem = DEADLINE - (time.time() - t0)
    if rem > 3.0:
        cp = best_pos.copy()
        ca = best_ang.copy()
        cs, _, _, _ = enclosing_side(cp, np.radians(ca))
        step_cd = 0.025
        min_cd = 0.0002
        t_cd = time.time()

        while time.time() - t_cd < min(rem * 0.55, 20.0) and step_cd > min_cd:
            improved = False
            for idx in range(N):
                if time.time() - t_cd > min(rem * 0.55, 20.0):
                    break
                for dim in range(3):
                    if time.time() - t_cd > min(rem * 0.55, 20.0):
                        break
                    for d in (1, -1):
                        np2 = cp.copy()
                        na2 = ca.copy()
                        if dim == 0:
                            np2[idx, 0] += d * step_cd
                        elif dim == 1:
                            np2[idx, 1] += d * step_cd
                        else:
                            na2[idx] += d * step_cd
                        ar2 = np.radians(na2)
                        ok2 = True
                        for j in range(N):
                            if j == idx:
                                continue
                            if not disjoint_fast(np2[idx, 0], np2[idx, 1], ar2[idx],
                                                 np2[j, 0], np2[j, 1], ar2[j]):
                                ok2 = False
                                break
                        if not ok2:
                            continue
                        ns2, _, _, _, _ = enclosing_side(np2, ar2)
                        if ns2 < cs - 1e-10:
                            cp, ca = np2, na2
                            cs = ns2
                            improved = True
                            break
            if not improved:
                step_cd *= 0.7
            else:
                step_cd *= 0.95

        if cs < best_side:
            best_side = cs
            best_pos = cp.copy()
            best_ang = ca.copy()
            best_side, best_phi, best_cx, best_cy = enclosing_side(best_pos, np.radians(best_ang))

    # --- Phase 3: Final Nelder-Mead ---
    rem2 = DEADLINE - (time.time() - t0)
    if rem2 > 5.0:
        def nm_obj(params):
            pos = np.zeros((N, 2))
            ang = np.zeros(N)
            for i in range(N):
                pos[i, 0] = params[3 * i]
                pos[i, 1] = params[3 * i + 1]
                ang[i] = params[3 * i + 2]
            ar = np.radians(ang)
            if not all_valid(pos, ar):
                return 1e6
            s, _, _, _ = enclosing_side(pos, ar)
            return s

        p0 = np.zeros(3 * N)
        for i in range(N):
            p0[3 * i] = best_pos[i, 0]
            p0[3 * i + 1] = best_pos[i, 1]
            p0[3 * i + 2] = best_ang[i]

        try:
            r = minimize(nm_obj, p0, method='Nelder-Mead',
                         options={'maxiter': 8000, 'xatol': 1e-9,
                                  'fatol': 1e-12, 'adaptive': True})
            if r.fun < best_side and r.fun < 100:
                np3 = np.zeros((N, 2))
                na3 = np.zeros(N)
                for i in range(N):
                    np3[i, 0] = r.x[3 * i]
                    np3[i, 1] = r.x[3 * i + 1]
                    na3[i] = r.x[3 * i + 2]
                if all_valid(np3, np.radians(na3)):
                    ns3, nphi3, ncx3, ncy3 = enclosing_side(np3, np.radians(na3))
                    if ns3 < best_side:
                        best_side = ns3
                        best_pos = np3
                        best_ang = na3
                        best_phi, best_cx, best_cy = nphi3, ncx3, ncy3
        except Exception:
            pass

    # Final safety check
    if not all_valid(best_pos, np.radians(best_ang)):
        for pi, ai in inits:
            if all_valid(pi, np.radians(ai)):
                best_pos = pi.copy()
                best_ang = ai.copy()
                best_side, best_phi, best_cx, best_cy = enclosing_side(best_pos, np.radians(best_ang))
                break
        else:
            best_pos, best_ang = init_honeycomb()
            best_side, best_phi, best_cx, best_cy = enclosing_side(best_pos, np.radians(best_ang))

    inner = [[float(best_pos[i, 0]), float(best_pos[i, 1]), float(best_ang[i])] for i in range(N)]
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    return inner, center, side, angle
