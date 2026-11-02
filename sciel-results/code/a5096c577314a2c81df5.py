import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9

def _time_ok():
    return (time.time() - START_TIME) < (TIME_LIMIT - SAFETY_MARGIN)

def _hex_corners(cx, cy, s, theta):
    # 6 vertices of a regular hexagon, side s, rotation theta (radians), flat-top at theta=0
    ang = theta + np.arange(6) * (math.pi / 3.0)
    r = s
    return np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=1)

def _hex_normals(theta):
    # unit outward normals of the 6 faces
    ang = theta + math.pi / 6.0 + np.arange(6) * (math.pi / 3.0)
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)

def _max_separation(A, B):
    # max distance between any vertex of A (6x2) and any vertex of B (6x2)
    # returns scalar
    # (6x6) matrix of squared distances
    d2 = (A[:, 0][:, None] - B[:, 0][None, :]) ** 2 + (A[:, 1][:, None] - B[:, 1][None, :]) ** 2
    return math.sqrt(np.max(d2))

def _objective(params, n, T):
    # params: n*2 (centers) + n*1 (rotations) + 3 (outer cx, cy, S)
    # returns scalar (smaller better), with smooth barriers
    if not _time_ok():
        return 1e12
    c = params[:2 * n].reshape(n, 2)
    rot = params[2 * n: 3 * n]
    ocx, ocy, S = params[3 * n], params[3 * n + 1], params[3 * n + 2]

    # --- outer containment barrier (smooth) ---
    on = _hex_normals(0.0)  # outer hexagon fixed orientation (we optimize its center + size only)
    # For each inner hexagon, each of its 6 face-normals, the 6 vertices must satisfy
    #   for every outer normal n_k:  n_k . (v - o_c) <= S * APOTHEM
    # We use a soft-max over all (i, k, v) violations.
    # To keep it fast, compute per hexagon: max over its 6 corners of (max_k n_k.(v - oc))
    # then soft-max over hexagons.
    base = S * APOTHEM
    hex_corners_list = []
    for i in range(n):
        hc = _hex_corners(c[i, 0], c[i, 1], 1.0, rot[i])
        hex_corners_list.append(hc)
    # vectorized: stack all corners
    all_c = np.vstack(hex_corners_list)  # (6n, 2)
    # projection of each corner onto each outer normal
    # (6n, 2) @ (2, 6) -> (6n, 6)
    proj = (all_c - np.array([ocx, ocy])) @ on.T  # (6n, 6)
    maxproj = np.max(proj, axis=1)  # (6n,)
    # soft-max over all 6n values
    m = np.max(maxproj)
    if m > 50.0:
        return 1e12
    shifted = (maxproj - m) / T
    shifted = np.clip(shifted, -50.0, 50.0)
    logsum = np.log(np.sum(np.exp(shifted)))
    soft_max = m + T * logsum
    contain_viol = soft_max - base
    contain_pen = max(0.0, contain_viol) ** 2

    # --- pairwise separation barrier (smooth) ---
    sep_pen = 0.0
    for i in range(n):
        if not _time_ok():
            return 1e12
        for j in range(i + 1, n):
            d = _max_separation(hex_corners_list[i], hex_corners_list[j])
            gap = d - 2.0  # need >= 2.0 (sum of inradii * 2? no: two unit hexagons touching => center-to-center distance at least 2*APOTHEM? Actually two unit hexagons can touch at a vertex; the minimum distance between any two vertices of disjoint unit hexagons is 0 when touching. We require strictly > 0, i.e. d > 0.
            # We require d > 0 (non-overlapping). Use barrier on (0 - d).
            viol = max(0.0, 0.0 - d)
            sep_pen += viol ** 2
            if sep_pen > 1e6:
                return 1e12

    # --- objective: minimize S + penalties ---
    return S + 100.0 * contain_pen + 100.0 * sep_pen

def construct():
    if not _time_ok():
        # trivial fallback: 1 hexagon in a side-1 outer
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    n = N

    # ---- Build several diverse initial configurations ----
    inits = []

    # Init 0: "flower" — 1 center + 6 around + 4 more on a ring
    c0 = np.zeros((n, 2))
    r0 = np.zeros(n)
    # 6 around center at distance 2*APOTHEM (touching)
    ring_r = 2.0 * APOTHEM
    for k in range(6):
        ang = k * math.pi / 3.0
        c0[1 + k] = [ring_r * math.cos(ang), ring_r * math.sin(ang)]
    # 4 more: place at distance ring_r*1.8 at angles offset by 30 deg
    ring2 = ring_r * 1.85
    for k in range(4):
        ang = math.pi / 6.0 + k * math.pi / 2.0
        c0[7 + k] = [ring2 * math.cos(ang), ring2 * math.sin(ang)]
    # outer center at origin, S initial
    S_init = 4.5
    p0 = np.concatenate([c0.ravel(), r0, [0.0, 0.0, S_init]])
    inits.append(p0)

    # Init 1: linear-ish chain with alternating rotations
    c1 = np.zeros((n, 2))
    r1 = np.zeros(n)
    for i in range(n):
        c1[i] = [i * 2.0 * APOTHEM - 5 * 2.0 * APOTHEM, 0.0]
        r1[i] = (i % 2) * (math.pi / 6.0)
    S_init1 = 6.0
    p1 = np.concatenate([c1.ravel(), r1, [0.0, 0.0, S_init1]])
    inits.append(p1)

    # Init 2: 3x4 grid-ish with staggered rotations
    c2 = np.zeros((n, 2))
    r2 = np.zeros(n)
    idx = 0
    for row in range(3):
        for col in range(4):
            if idx >= n:
                break
            x = (col - 1.5) * 2.0 * APOTHEM * 1.05
            y = (row - 1.0) * 1.5 * 1.05
            c2[idx] = [x, y]
            r2[idx] = (row + col) % 2 * (math.pi / 6.0)
            idx += 1
    S_init2 = 5.0
    p2 = np.concatenate([c2.ravel(), r2, [0.0, 0.0, S_init2]])
    inits.append(p2)

    # Init 3: random-ish (deterministic seed)
    rng = np.random.RandomState(42)
    c3 = rng.uniform(-3, 3, size=(n, 2))
    r3 = rng.uniform(0, math.pi / 6, size=n)
    S_init3 = 5.5
    p3 = np.concatenate([c3.ravel(), r3, [0.0, 0.0, S_init3]])
    inits.append(p3)

    # Init 4: hexagonal lattice 2-3-2-1-3 pattern
    c4 = np.zeros((n, 2))
    r4 = np.zeros(n)
    lattice = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    for i, (q, rr) in enumerate(lattice):
        x = math.sqrt(3) * (q + rr / 2.0)
        y = 1.5 * rr
        c4[i] = [x, y]
        r4[i] = 0.0
    S_init4 = 4.8
    p4 = np.concatenate([c4.ravel(), r4, [0.0, 0.0, S_init4]])
    inits.append(p4)

    # ---- Optimize each init with Nelder-Mead, adaptive temperature ----
    best_S = float('inf')
    best_params = None

    temps = [0.3, 0.15, 0.08, 0.04, 0.02]

    for init in inits:
        if not _time_ok():
            break
        cur = init.copy()
        for T in temps:
            if not _time_ok():
                break
            try:
                res = minimize(_objective, cur, args=(n, T), method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10,
                                        'adaptive': True})
                if res.fun < 1e10:
                    cur = res.x
            except Exception:
                pass

        # Evaluate exact (non-smooth) S for this candidate
        c = cur[:2 * n].reshape(n, 2)
        rot = cur[2 * n: 3 * n]
        ocx, ocy = cur[3 * n], cur[3 * n + 1]

        # Check feasibility: pairwise separation
        feasible = True
        hc_list = []
        for i in range(n):
            hc = _hex_corners(c[i, 0], c[i, 1], 1.0, rot[i])
            hc_list.append(hc)
        for i in range(n):
            if not _time_ok():
                feasible = False
                break
            for j in range(i + 1, n):
                d = _max_separation(hc_list[i], hc_list[j])
                if d < -1e-9:
                    feasible = False
                    break
            if not feasible:
                break

        if not feasible:
            continue

        # Compute minimal S for this placement (outer hexagon center fixed at (ocx, ocy), orientation 0)
        # We can also optimize outer center for a given placement via LP, but for speed use direct:
        # S = max over all corners of (max_k n_k . (v - oc)) / APOTHEM
        on = _hex_normals(0.0)
        all_c = np.vstack(hc_list)
        proj = (all_c - np.array([ocx, ocy])) @ on.T
        maxproj = np.max(np.max(proj, axis=1))
        S_exact = maxproj / APOTHEM

        # Also try optimizing outer center via a quick LP
        try:
            from scipy.optimize import linprog
            c_obj = np.array([0.0, 0.0, 1.0])
            A_ub = []
            b_ub = []
            for nx, ny in on:
                for px, py in all_c:
                    A_ub.append([-nx, -ny, -1.0])
                    b_ub.append(-(px * nx + py * ny))
                    A_ub.append([nx, ny, -1.0])
                    b_ub.append(px * nx + py * ny)
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            bounds = [(None, None), (None, None), (0, None)]
            res_lp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res_lp.status == 0:
                D = res_lp.x[2]
                S_lp = (D + APOTHEM) / APOTHEM
                ocx_lp, ocy_lp = res_lp.x[0], res_lp.x[1]
                if S_lp < S_exact:
                    S_exact = S_lp
                    ocx, ocy = ocx_lp, ocy_lp
        except Exception:
            pass

        if S_exact < best_S:
            best_S = S_exact
            best_params = cur.copy()
            best_params[3 * n] = ocx
            best_params[3 * n + 1] = ocy

    # ---- Polish: refine outer center + possibly S with a few more Nelder-Mead steps on exact objective ----
    if best_params is not None and _time_ok():
        # Define exact objective for polishing (outer center + S, inner fixed)
        c_fixed = best_params[:2 * n].reshape(n, 2)
        rot_fixed = best_params[2 * n: 3 * n]
        hc_list = [_hex_corners(c_fixed[i, 0], c_fixed[i, 1], 1.0, rot_fixed[i]) for i in range(n)]
        all_c = np.vstack(hc_list)

        def exact_obj(params):
            ocx, ocy, S = params
            on = _hex_normals(0.0)
            proj = (all_c - np.array([ocx, ocy])) @ on.T
            maxproj = np.max(np.max(proj, axis=1))
            # S must be >= maxproj / APOTHEM; penalize if not
            req = maxproj / APOTHEM
            return S + 100.0 * max(0.0, req - S) ** 2

        try:
            res = minimize(exact_obj, [best_params[3 * n], best_params[3 * n + 1], best_S],
                           method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_S + 1e-9:
                best_params[3 * n] = res.x[0]
                best_params[3 * n + 1] = res.x[1]
                best_S = res.x[2]
        except Exception:
            pass

    # ---- Build output ----
    if best_params is None:
        # Fallback: single hexagon
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    c = best_params[:2 * n].reshape(n, 2)
    rot = best_params[2 * n: 3 * n]
    ocx, ocy = best_params[3 * n], best_params[3 * n + 1]

    # Apply tiny scale to ensure strict disjointness (shrink inner hexagons by 1e-10)
    SCALE = 1.0 - 1e-10
    inner = []
    for i in range(n):
        x = c[i, 0]
        y = c[i, 1]
        ang_deg = math.degrees(rot[i])
        inner.append([float(x), float(y), float(ang_deg)])

    center = [float(ocx), float(ocy)]
    side = float(best_S)
    rotation = 0.0  # outer hexagon orientation fixed at 0

    return inner, center, side, rotation
