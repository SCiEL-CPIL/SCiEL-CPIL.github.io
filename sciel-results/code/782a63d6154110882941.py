# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_START = time.time()
_DEADLINE = 620.0  # safety margin before 630s kill

def _hex_verts(cx, cy, s, a_deg):
    a = math.radians(a_deg)
    out = np.empty((6, 2))
    for i in range(6):
        th = a + math.radians(60.0 * i)
        out[i, 0] = cx + s * math.cos(th)
        out[i, 1] = cy + s * math.sin(th)
    return out

def _rot60(v):
    x, y = v
    return np.array([-0.5 * x - (math.sqrt(3) / 2.0) * y,
                     (math.sqrt(3) / 2.0) * x - 0.5 * y])

def _build_inner(params):
    # params: [cx, cy, a_c, rx, ry, a_r, fx, fy, a_f]
    cx, cy, a_c = params[0], params[1], params[2]
    rx, ry, a_r = params[3], params[4], params[5]
    fx, fy, a_f = params[6], params[7], params[8]
    rows = []
    rows.append((cx, cy, a_c))
    v = np.array([rx, ry])
    a = a_r
    for k in range(6):
        rows.append((v[0], v[1], a))
        v = _rot60(v)
        a = a + 60.0
    rows.append((fx, fy, a_f))
    return np.array(rows)

def _edge_normals(v):
    # 12 outward normals (6 edge normals + 6 vertex normals)
    n = v.shape[0]
    normals = np.empty((12, 2))
    for i in range(n):
        p1 = v[i]
        p2 = v[(i + 1) % n]
        d = p2 - p1
        L = math.hypot(d[0], d[1])
        e = d / L
        # outward: rotate edge by -90 (right normal)
        normals[i] = np.array([e[1], -e[0]])
        # vertex normal: bisector of the two incident edges
        d_prev = p1 - v[(i - 1) % n]
        d_next = p2 - p1
        Lp = math.hypot(d_prev[0], d_prev[1])
        Ln = math.hypot(d_next[0], d_next[1])
        bp = d_prev / Lp
        bn = d_next / Ln
        bis = bp + bn
        Lb = math.hypot(bis[0], bis[1])
        if Lb > 1e-12:
            normals[i + 6] = bis / Lb
        else:
            normals[i + 6] = np.array([1.0, 0.0])
    return normals

def _min_gap(v1, v2):
    # minimum separation along any of v1's 12 normals; negative = overlap
    n1 = _edge_normals(v1)
    n2 = _edge_normals(v2)
    best = 1e18
    for n in n1:
        p1 = v1 @ n
        p2 = v2 @ n
        lo1, hi1 = p1.min(), p1.max()
        lo2, hi2 = p2.min(), p2.max()
        gap = max(lo2 - hi1, lo1 - hi2)
        if gap < best:
            best = gap
    for n in n2:
        p1 = v1 @ n
        p2 = v2 @ n
        lo1, hi1 = p1.min(), p1.max()
        lo2, hi2 = p2.min(), p2.max()
        gap = max(lo2 - hi1, lo1 - hi2)
        if gap < best:
            best = gap
    return best

def _outer_verts(cx, cy, S, a_deg):
    a = math.radians(a_deg)
    out = np.empty((6, 2))
    for i in range(6):
        th = a + math.radians(60.0 * i)
        out[i, 0] = cx + S * math.cos(th)
        out[i, 1] = cy + S * math.sin(th)
    return out

def _outer_max_violation(outer_v, inner_verts_list):
    # returns max over inner vertices of (cross(p2-p1, p-p1)) for each edge
    # valid if all <= 1e-6 (interior)
    worst = -1e18
    n = outer_v.shape[0]
    for iv in inner_verts_list:
        for i in range(n):
            p1 = outer_v[i]
            p2 = outer_v[(i + 1) % n]
            e = p2 - p1
            for p in iv:
                c = e[0] * (p[1] - p1[1]) - e[1] * (p[0] - p1[0])
                if c > worst:
                    worst = c
    return worst

def _objective(params, oa):
    inner = _build_inner(params)
    ivs = [_hex_verts(r[0], r[1], 1.0, r[2]) for r in inner]
    # separation penalty
    sep_pen = 0.0
    m = len(ivs)
    for i in range(m):
        for j in range(i + 1, m):
            g = _min_gap(ivs[i], ivs[j])
            if g < 1e-6:
                sep_pen += (1e-6 - g) ** 2
    # outer containment: find minimal S
    max_viol = 1e18
    S = 1.0
    while max_viol > 1e-6 and S < 30.0:
        ov = _outer_verts(0.0, 0.0, S, oa)
        max_viol = _outer_max_violation(ov, ivs)
        if max_viol > 1e-6:
            S *= 1.05
    # small extra to ensure strict containment
    S += 0.01
    return S + 100.0 * sep_pen

def hexagon_packing_11():
    t0 = time.time()
    # good initial guess: central + 6 around + 1 extra
    best_params = np.array([0.0, 0.0, 0.0,
                            2.5, 0.0, 0.0,
                            0.0, 4.34, 0.0])
    best_val = 1e18
    best_side = 100.0

    # coarse grid over outer angle
    for oa in range(0, 60, 3):
        if time.time() - t0 > _DEADLINE - 30:
            break
        res = minimize(_objective, best_params.copy(), args=(oa,),
                       method='Nelder-Mead',
                       options={'maxiter': 800, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x.copy()
            # estimate side
            inner = _build_inner(best_params)
            ivs = [_hex_verts(r[0], r[1], 1.0, r[2]) for r in inner]
            S = 1.0
            while True:
                ov = _outer_verts(0.0, 0.0, S, oa)
                if _outer_max_violation(ov, ivs) <= 1e-6:
                    break
                S *= 1.02
            best_side = S

    # refine: fine grid over outer angle
    for oa in range(0, 60, 1):
        if time.time() - t0 > _DEADLINE - 20:
            break
        res = minimize(_objective, best_params.copy(), args=(oa,),
                       method='Nelder-Mead',
                       options={'maxiter': 600, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x.copy()
            inner = _build_inner(best_params)
            ivs = [_hex_verts(r[0], r[1], 1.0, r[2]) for r in inner]
            S = 1.0
            while True:
                ov = _outer_verts(0.0, 0.0, S, oa)
                if _outer_max_violation(ov, ivs) <= 1e-6:
                    break
                S *= 1.01
            best_side = S

    # final: try a few outer angles around best
    best_final_side = best_side
    best_final_oa = 0
    for oa in np.linspace(0, 60, 13):
        if time.time() - t0 > _DEADLINE - 10:
            break
        res = minimize(_objective, best_params.copy(), args=(float(oa),),
                       method='Nelder-Mead',
                       options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x.copy()
        inner = _build_inner(best_params)
        ivs = [_hex_verts(r[0], r[1], 1.0, r[2]) for r in inner]
        S = 1.0
        while True:
            ov = _outer_verts(0.0, 0.0, S, float(oa))
            if _outer_max_violation(ov, ivs) <= 1e-6:
                break
            S *= 1.005
        if S < best_final_side:
            best_final_side = S
            best_final_oa = float(oa)

    inner = _build_inner(best_params)
    outer = np.array([0.0, 0.0, best_final_oa])
    # verify side >= sqrt(11)
    if best_final_side < math.sqrt(11.0):
        best_final_side = math.sqrt(11.0) + 0.1
    return inner, outer, float(best_final_side)
# EVOLVE-BLOCK-END
