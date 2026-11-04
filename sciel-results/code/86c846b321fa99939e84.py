"""Rewritten program: Differential Evolution global search + exact polygon containment."""
import time
import numpy as np

N = 11

def _hex_verts(cx, cy, s, ang):
    a = np.deg2rad(ang)
    th = a + np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    return np.stack([cx + s*np.cos(th), cy + s*np.sin(th)], axis=1)

def _cross(a, b):
    return a[...,0]*b[...,1] - a[...,1]*b[...,0]

def _point_in_poly(px, py, poly):
    n = len(poly)
    c = 0.0
    for i in range(n):
        x1, y1 = poly[i][0], poly[i][1]
        x2, y2 = poly[(i+1)%n][0], poly[(i+1)%n][1]
        if (y1 > py) != (y2 > py):
            xint = (x2-x1)*(py-y1)/(y2-y1) + x1
            if xint > px:
                c += 1
    return c % 2 == 1

def _poly_in_poly(inner, outer):
    for i in range(len(inner)):
        if not _point_in_poly(inner[i][0], inner[i][1], outer):
            return False
    return True

def _poly_overlap(a, b):
    for poly in (a, b):
        for i in range(len(poly)):
            p1 = poly[i]; p2 = poly[(i+1)%len(poly)]
            d = p2 - p1
            nrm = np.array([-d[1], d[0]])
            proj_a = nrm @ (a - p1)
            proj_b = nrm @ (b - p1)
            if proj_a.max() < proj_b.min() or proj_b.max() < proj_a.min():
                return False
    return True

def _fits(x, outer_verts):
    for i in range(len(x)-1):
        v = _hex_verts(x[2*i], x[2*i+1], 1.0, x[2*i+2])
        if _poly_in_poly(v, outer_verts):
            for j in range(i+1, len(x)-1):
                w = _hex_verts(x[2*j], x[2*j+1], 1.0, x[2*j+2])
                if _poly_overlap(v, w):
                    return False
    return True

def _objective(x, outer_verts):
    if _fits(x, outer_verts):
        return 0.0
    return 1.0

def construct():
    t0 = time.time()
    DEADLINE = 140.0
    best_x = None
    best_side = float('inf')
    rng = np.random.default_rng(12345)

    # initial guess: horizontal row
    init = np.zeros(3*N)
    for k in range(N):
        init[2*k] = 4.0*(k - (N-1)//2)
        init[2*k+1] = 0.0
        init[2*k+2] = 0.0

    def _check(side, oang, x, outer_verts, t0):
        nonlocal best_x, best_side
        if _fits(x, outer_verts):
            if side < best_side:
                best_side = side
                best_x = x.copy()
            return True
        return False

    # Phase 1: optimize outer rotation for a fixed row layout
    for oang in np.linspace(0, 60, 25):
        if time.time() > DEADLINE: break
        # find minimal side via bisection
        lo, hi = 18.0, 45.0
        for _ in range(28):
            mid = 0.5*(lo+hi)
            ov = _hex_verts(0.0, 0.0, mid, oang)
            if _fits(init, ov):
                hi = mid
            else:
                lo = mid
        if hi < best_side:
            best_side = hi
            best_x = init.copy()
            best_oang = oang
            best_oval = (0.0, 0.0)

    # Phase 2: Differential Evolution to optimize positions + outer
    if time.time() < DEADLINE - 5:
        # variables: positions (3N) + outer angle (1) = 3N+1
        # side is minimized via bisection after DE
        def _de_obj(xv):
            x = xv[:3*N]
            oang = xv[3*N] % 60.0
            # quick side estimate: use a generous side, check fit, return penalty
            side_est = 26.0
            ov = _hex_verts(0.0, 0.0, side_est, oang)
            if _fits(x, ov):
                return 0.0
            # penalty: count overlaps + out-of-bounds
            pen = 0.0
            for i in range(N):
                v = _hex_verts(x[2*i], x[2*i+1], 1.0, x[2*i+2])
                for p in v:
                    if not _point_in_poly(p[0], p[1], ov):
                        pen += 1.0
                for j in range(i+1, N):
                    w = _hex_verts(x[2*j], x[2*j+1], 1.0, x[2*j+2])
                    if _poly_overlap(v, w):
                        pen += 10.0
            return pen

        # run a lightweight DE
        pop_size = 12
        dim = 3*N + 1
        pop = [init.copy() + rng.normal(0, 3.0, dim) for _ in range(pop_size)]
        pop[0] = init.copy()
        pop[0][3*N] = best_oang
        F = 0.8; CR = 0.8
        for gen in range(60):
            if time.time() > DEADLINE - 5: break
            for i in range(pop_size):
                a, b, c = rng.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                trial = pop[a] + F*(pop[b] - pop[c])
                for d in range(dim):
                    if rng.random() < CR:
                        trial[d] = pop[c][d]
                # wrap angle
                trial[3*N] = trial[3*N] % 60.0
                if _de_obj(trial) < _de_obj(pop[i]):
                    pop[i] = trial
            # after DE, for each candidate find min side via bisection
            for i in range(pop_size):
                if time.time() > DEADLINE - 5: break
                x = pop[i][:3*N]
                oang = pop[i][3*N]
                lo, hi = 18.0, 45.0
                ov_hi = _hex_verts(0.0, 0.0, hi, oang)
                if not _fits(x, ov_hi):
                    continue
                for _ in range(22):
                    mid = 0.5*(lo+hi)
                    ov = _hex_verts(0.0, 0.0, mid, oang)
                    if _fits(x, ov):
                        hi = mid
                    else:
                        lo = mid
                if hi < best_side:
                    best_side = hi
                    best_x = x.copy()
                    best_oang = oang
                    best_oval = (0.0, 0.0)

    if best_x is None:
        best_x = init
        best_side = 40.0
        best_oang = 0.0
        best_oval = (0.0, 0.0)

    inner = [[best_x[2*i], best_x[2*i+1], best_x[2*i+2]] for i in range(N)]
    return inner, best_oval, best_side, best_oang
