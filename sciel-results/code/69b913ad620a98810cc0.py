"""Smooth-continuous hexagon packing: Nelder-Mead on min-enclosing-hex side + quadratic overlap penalty."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 135.0  # safety margin before 150s kill

def _hex_verts(cx, cy, s, ang_deg):
    a = np.radians(ang_deg)
    th = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])

def _min_side(all_pts, cx, cy, outer_ang_deg):
    """Min outer side S so all_pts inside hex(cx,cy,S,outer_ang)."""
    if len(all_pts) == 0:
        return 0.0
    dx = all_pts[:, 0] - cx
    dy = all_pts[:, 1] - cy
    a = np.radians(outer_ang_deg)
    ca, sa = np.cos(a), np.sin(a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    s3 = np.sqrt(3.0)
    c = np.stack([lx, -lx, lx + ly / s3, -lx + ly / s3, -lx - ly / s3, lx - ly / s3], axis=1)
    return float(c.max())

def _all_vertices(positions, angles, side=1.0):
    n = len(positions)
    pos = np.asarray(positions, dtype=np.float64)
    ang = np.radians(np.asarray(angles, dtype=np.float64))
    base = np.arange(6) * np.pi / 3.0
    ta = ang[:, None] + base[None, :]
    vx = pos[:, 0:1] + side * np.cos(ta)
    vy = pos[:, 1:1] + side * np.sin(ta)
    return np.column_stack([vx.ravel(), vy.ravel()])

def _separation(v1, v2, tol=1e-9):
    """Return max gap (positive if separated) between two convex hexagons via SAT."""
    axes = []
    for v in (v1, v2):
        for i in range(6):
            e = v[(i + 1) % 6] - v[i]
            nrm = np.hypot(e[0], e[1])
            if nrm > 1e-12:
                axes.append([e[1] / nrm, -e[0] / nrm])
    best_gap = 0.0  # we want max over axes of (min_other - max_self)
    for ax in axes:
        p1 = v1 @ ax
        p2 = v2 @ ax
        gap = min(p1.min() - p2.max(), p2.min() - p1.max())
        if gap > best_gap:
            best_gap = gap
    return best_gap  # negative => overlap depth

def _overlap_penalty(positions, angles, w=50.0, power=2):
    n = len(positions)
    verts = [_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(n)]
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sep = _separation(verts[i], verts[j])
            if sep < 0:
                d = -sep  # penetration depth
                pen += w * (d ** power)
    return pen

def _objective_flat(params, outer_cx=0.0, outer_cy=0.0, outer_ang=0.0):
    pos = params[0:2 * N:2]
    ang = params[1:2 * N:2]
    all_pts = _all_vertices(pos, ang, 1.0)
    side = _min_side(all_pts, outer_cx, outer_cy, outer_ang)
    pen = _overlap_penalty(list(zip(pos, np.zeros(N))) if False else np.column_stack([pos[0::2], pos[1::2]]), ang)
    return side + pen

def _make_initial(sym_type=0):
    """Return initial 33-vector params [x0,y0,ang0, x1,y1,ang1, ...]."""
    s3 = np.sqrt(3.0)
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    if sym_type == 0:
        # hex lattice: center + 6 neighbors + 4 ring-2
        pos[0] = [0, 0]
        for i in range(1, 7):
            a = (i - 1) * np.pi / 3
            pos[i] = [2.0 * np.cos(a), 2.0 * np.sin(a)]
        for k, a in enumerate([0, np.pi / 3, 2 * np.pi / 3, np.pi]):
            pos[7 + k] = [4.0 * np.cos(a), 4.0 * np.sin(a)]
    elif sym_type == 1:
        # 3-fold symmetric: center + 2 rings of 3
        pos[0] = [0, 0]
        for i in range(1, 4):
            a = (i - 1) * 2 * np.pi / 3
            pos[i] = [2.0 * np.cos(a), 2.0 * np.sin(a)]
        for i in range(4, 7):
            a = (i - 4) * 2 * np.pi / 3 + np.pi / 3
            pos[i] = [2.0 * np.cos(a), 2.0 * np.sin(a)]
        for i in range(7, 11):
            a = (i - 7) * 2 * np.pi / 4 + np.pi / 4
            pos[i] = [3.8 * np.cos(a), 3.8 * np.sin(a)]
    elif sym_type == 2:
        # 6-fold: center + 6 + 4
        pos[0] = [0, 0]
        for i in range(1, 7):
            a = (i - 1) * np.pi / 3
            pos[i] = [2.0 * np.cos(a), 2.0 * np.sin(a)]
        for k in range(4):
            a = k * np.pi / 2
            pos[7 + k] = [4.0 * np.cos(a), 4.0 * np.sin(a)]
    elif sym_type == 3:
        # random-ish spread
        rng = np.random.RandomState(7 + sym_type)
        pos = rng.uniform(-3, 3, size=(N, 2))
        # ensure center is at origin
        pos[0] = [0, 0]
        ang = rng.uniform(-15, 15, size=N)
    else:
        # linear row (fallback)
        for i in range(N):
            pos[i] = [4.0 * (i - N // 2), 0.0]
    params = np.zeros(3 * N)
    for i in range(N):
        params[3 * i] = pos[i, 0]
        params[3 * i + 1] = pos[i, 1]
        params[3 * i + 2] = ang[i]
    return params

def _eval_params(params):
    """Return (side, penalty, is_valid)."""
    pos = np.column_stack([params[0::3], params[1::3]])
    ang = params[2::3]
    all_pts = _all_vertices(pos, ang, 1.0)
    side = _min_side(all_pts, 0.0, 0.0, 0.0)
    pen = _overlap_penalty(pos, ang)
    # validity: no overlap AND all inside
    valid = pen < 1e-7
    if valid:
        # check containment
        outer_verts = _hex_verts(0.0, 0.0, side, 0.0)
        # quick check: all inner verts inside outer
        # Use point-in-hex
        s3 = np.sqrt(3.0)
        dx = all_pts[:, 0]
        dy = all_pts[:, 1]
        # For angle 0 hexagon: constraints |lx|<=S, |lx+ly/s3|<=S, |-lx+ly/s3|<=S  (lx=dx, ly=dy for ang 0)
        # Actually for outer angle 0: lx=dx, ly=dy
        inside = (np.abs(dx) <= side + 1e-9) & \
                 (np.abs(dx + dy / s3) <= side + 1e-9) & \
                 (np.abs(-dx + dy / s3) <= side + 1e-9)
        valid = bool(inside.all())
    return side, pen, valid

def construct():
    global START_TIME, DEADLINE
    best_params = None
    best_side = 1e9
    best_valid = False

    # Try multiple initializations with Nelder-Mead
    n_starts = 5
    for start_idx in range(n_starts):
        if time.time() > DEADLINE - 15.0:
            break
        p0 = _make_initial(sym_type=start_idx)
        # Add small jitter for diversity except first
        if start_idx > 0:
            rng = np.random.RandomState(100 + start_idx)
            p0 = p0 + rng.normal(0, 0.1, size=3 * N)

        def obj(params):
            if time.time() > DEADLINE - 10.0:
                return 1e6
            pos = np.column_stack([params[0::3], params[1::3]])
            ang = params[2::3]
            all_pts = _all_vertices(pos, ang, 1.0)
            side = _min_side(all_pts, 0.0, 0.0, 0.0)
            pen = _overlap_penalty(pos, ang)
            return side + pen

        try:
            res = minimize(
                obj, p0, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            p_best = res.x
            s, pen, valid = _eval_params(p_best)
            if valid and s < best_side:
                best_side = s
                best_params = p_best.copy()
                best_valid = True
            elif not best_valid and pen < 1e-3 and s < best_side:
                # Accept near-valid as fallback
                best_side = s
                best_params = p_best.copy()
        except Exception:
            pass

    # If no valid found, do a final polish on best_params
    if best_params is not None and time.time() < DEADLINE - 10.0:
        def obj2(params):
            pos = np.column_stack([params[0::3], params[1::3]])
            ang = params[2::3]
            all_pts = _all_vertices(pos, ang, 1.0)
            side = _min_side(all_pts, 0.0, 0.0, 0.0)
            pen = _overlap_penalty(pos, ang, w=200.0, power=2)
            return side + pen
        try:
            res = minimize(obj2, best_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            p_pol = res.x
            s, pen, valid = _eval_params(p_pol)
            if valid and s < best_side:
                best_side = s
                best_params = p_pol
                best_valid = True
        except Exception:
            pass

    # Build output
    if best_params is not None:
        pos = np.column_stack([best_params[0::3], best_params[1::3]])
        ang = best_params[2::3]
        inner_list = [[float(pos[i, 0]), float(pos[i, 1]), float(ang[i])] for i in range(N)]
        # Recompute exact side
        all_pts = _all_vertices(pos, ang, 1.0)
        final_side = _min_side(all_pts, 0.0, 0.0, 0.0)
        # Final validity check
        s, pen, valid = _eval_params(best_params)
        if not valid:
            # Fallback to a safe linear arrangement
            inner_list = [[4.0 * (i - N // 2), 0.0, 0.0] for i in range(N)]
            final_side = 4.0 * N
        return inner_list, (0.0, 0.0), float(final_side), 0.0
    else:
        # Last resort
        inner_list = [[4.0 * (i - N // 2), 0.0, 0.0] for i in range(N)]
        return inner_list, (0.0, 0.0), 4.0 * N, 0.0
