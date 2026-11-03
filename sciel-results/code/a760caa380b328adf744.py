"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def _hex_support(u_angle, theta):
    # support function of unit hexagon (circumradius 1) at rotation theta, in direction u_angle
    # vertices at theta + 60*m degrees
    best = -1.0
    for m in range(6):
        a = theta + math.radians(60.0 * m)
        c = math.cos(u_angle - a)
        if c > best:
            best = c
    return best

def _hex_edge_normals(theta):
    # 6 outward edge-normal angles of unit hexagon at rotation theta
    return [theta + math.radians(30.0 + 60.0 * m) for m in range(6)]

def _outer_normals():
    # outer hexagon fixed at rotation 0, center origin; edge normals at 30+60m
    return [math.radians(30.0 + 60.0 * m) for m in range(6)]

def _objective_and_penalty(params):
    # params: 3N floats: x_i, y_i, theta_i for i in 0..N-1
    # returns (objective, penalty)
    S = 1.0
    pen = 0.0
    centers = []
    angles = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        th = params[3*i+2]
        centers.append((x, y))
        angles.append(th)
    on = _outer_normals()
    for k, nk in enumerate(on):
        mx = -1e9
        for i in range(N):
            x, y = centers[i]
            th = angles[i]
            h = _hex_support(nk, th)
            val = x*math.cos(nk) + y*math.sin(nk) + h
            if val > mx:
                mx = val
        # apothem = S*sqrt3/2 ; need mx <= S*sqrt3/2 -> S >= mx*2/sqrt3
        S_k = mx * 2.0 / math.sqrt(3.0)
        if S_k > S:
            S = S_k
    # disjointness penalty
    for i in range(N):
        for j in range(i+1, N):
            xi, yi = centers[i]
            xj, yj = centers[j]
            d = (xi - xj, yi - yj)
            thi = angles[i]
            thj = angles[j]
            sep = False
            # check 12 directions
            for nk in _hex_edge_normals(thi):
                ux = math.cos(nk); uy = math.sin(nk)
                proj = d[0]*ux + d[1]*uy
                hi = _hex_support(nk, thi)
                hj = _hex_support(nk, thj)
                if proj >= hi + hj - 1e-9:
                    sep = True
                    break
            if not sep:
                for nk in _hex_edge_normals(thj):
                    ux = math.cos(nk); uy = math.sin(nk)
                    proj = d[0]*ux + d[1]*uy
                    hi = _hex_support(nk, thi)
                    hj = _hex_support(nk, thj)
                    if proj >= hi + hj - 1e-9:
                        sep = True
                        break
            if not sep:
                # compute overlap amount: max over the 12 dirs of (hi+hj - proj)
                worst = -1e9
                for nk in _hex_edge_normals(thi) + _hex_edge_normals(thj):
                    ux = math.cos(nk); uy = math.sin(nk)
                    proj = d[0]*ux + d[1]*uy
                    hi = _hex_support(nk, thi)
                    hj = _hex_support(nk, thj)
                    ov = hi + hj - proj
                    if ov > worst:
                        worst = ov
                pen += max(0.0, worst)
    return S, pen

def _total_penalty(params):
    S, pen = _objective_and_penalty(params)
    return S + 1000.0 * pen

def construct():
    t0 = time.time()
    deadline = t0 + 120.0
    best_params = None
    best_score = float('inf')
    best_S = float('inf')
    best_pen = float('inf')

    # Try several initial configurations
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    from scipy.optimize import minimize

    def _init_config():
        # ring of 12 hexagons around center
        params = []
        for i in range(N):
            ang = math.radians(360.0 * i / N)
            r = 1.5
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            th = math.radians(360.0 * i / N)  # align
            params.extend([x, y, th])
        return params

    def _init_random():
        params = []
        for i in range(N):
            x = (np_rng.random() - 0.5) * 4.0
            y = (np_rng.random() - 0.5) * 4.0
            th = math.radians(60.0 * np_rng.integers(0, 6))
            params.extend([x, y, th])
        return params

    def _eval(params):
        S, pen = _objective_and_penalty(params)
        return S, pen

    # Start with ring
    starts = [_init_config()]
    for _ in range(5):
        starts.append(_init_random())

    for s, p0 in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
        try:
            res = minimize(_total_penalty, p0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-5})
            p = res.x
            S, pen = _eval(p)
            if pen < 1e-6 and S < best_S:
                best_S = S
                best_pen = pen
                best_params = list(p)
        except Exception:
            pass

    # Local refinement with smaller steps if we have a good candidate
    if best_params is not None:
        if time.time() < deadline - 15.0:
            try:
                res = minimize(_total_penalty, best_params, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-6,
                                        'initial_simplex': 0.01})
                p = res.x
                S, pen = _eval(p)
                if pen < 1e-6 and S < best_S:
                    best_S = S
                    best_pen = pen
                    best_params = list(p)
            except Exception:
                pass

    # If no valid found, do a simple greedy placement
    if best_params is None:
        # fallback: place in a hex grid
        params = []
        # hex grid positions for 12 hexagons
        # axial coordinates
        axial = []
        for q in range(-3, 4):
            for r in range(-3, 4):
                if abs(q + r) <= 3:
                    axial.append((q, r))
        # take 12
        axial = axial[:12]
        for (q, r) in axial:
            x = 1.5 * q
            y = math.sqrt(3)/2 * (r + q/2)
            th = 0.0
            params.extend([x, y, th])
        best_params = params
        best_S, best_pen = _eval(params)

    # Build inner list
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        th_deg = math.degrees(best_params[3*i+2]) % 360.0
        inner.append([float(x), float(y), float(th_deg)])

    return inner, (0.0, 0.0), float(best_S), 0.0
# EVOLVE-BLOCK-END

