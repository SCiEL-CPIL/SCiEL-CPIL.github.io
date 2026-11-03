import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 115.0
_T0 = time.time()

# ---- geometry helpers ----
def _hex_verts(cx, cy, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = np.empty((6, 2))
    for i in range(6):
        t = math.radians(60.0 * i)
        x, y = math.cos(t), math.sin(t)
        out[i, 0] = cx + ca * x - sa * y
        out[i, 1] = cy + sa * x + ca * y
    return out

def _rot(v, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    x, y = v[0], v[1]
    return np.array([ca * x - sa * y, sa * x + ca * y])

def _sdf_hex(point, cx, cy, ang_deg):
    """Signed distance from point to a unit regular hexagon (flat-top, rot ang_deg).
    Positive outside, negative inside, 0 on boundary. Smooth approximation of the
    exact hexagon distance (corner arcs clipped to hexagon radius)."""
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    dx, dy = point[0] - cx, point[1] - cy
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    ax, ay = abs(lx), abs(ly)
    if ax < 0.5 and ay < math.sqrt(3.0) / 2.0:
        return -min(0.5 - ax, math.sqrt(3.0) / 2.0 - ay)
    # closest point on hexagon boundary (flat-top)
    if ay > math.sqrt(3.0) / 2.0:
        qx, qy = lx, math.copysign(math.sqrt(3.0) / 2.0, ly)
    elif math.sqrt(3.0) * ax > 3.0 * ay:
        qx, qy = math.copysign(0.5, lx), math.copysign(ay, ly)
    else:
        s = 1.0 / (math.sqrt(3.0) * ax + ay)
        qx, qy = math.copysign(math.sqrt(3.0) / 2.0 * s, lx), math.copysign(0.5 * s, ly)
    return math.hypot(lx - qx, ly - qy)

def _outer_sdf(point, cx, cy, S, ang_deg):
    """Signed distance from point to outer hexagon of side S (center cx,cy, rot ang_deg)."""
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    dx, dy = point[0] - cx, point[1] - cy
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    ax, ay = abs(lx), abs(ly)
    R = S
    if ax < R and ay < math.sqrt(3.0) * R / 2.0:
        return -min(R - ax, math.sqrt(3.0) * R / 2.0 - ay)
    if ay > math.sqrt(3.0) * R / 2.0:
        qx, qy = lx, math.copysign(math.sqrt(3.0) * R / 2.0, ly)
    elif math.sqrt(3.0) * ax > 3.0 * ay:
        qx, qy = math.copysign(R, lx), math.copysign(ay, ly)
    else:
        s = 1.0 / (math.sqrt(3.0) * ax + ay)
        qx, qy = math.copysign(math.sqrt(3.0) / 2.0 * s, lx), math.copysign(0.5 * s, ly)
    return math.hypot(lx - qx, ly - qy)

def _min_pair_gap(P, A):
    """Smooth approximation of min over all pairs of (gap between hex i and hex j)."""
    m = len(P)
    best = -1e9
    for i in range(m):
        pi = P[i]
        for j in range(i + 1, m):
            pj = P[j]
            d = math.hypot(pi[0] - pj[0], pi[1] - pj[1])
            if d < 1.4:
                continue
            ux = (pj[0] - pi[0]) / d
            uy = (pj[1] - pi[1]) / d
            ang = A[i] + math.degrees(math.atan2(uy, ux))
            gap = _sdf_hex(pj, pi[0], pi[1], ang) + 1.0
            if gap > best:
                best = gap
    return best

def _min_inner_outer(P, A, O):
    """Smooth approximation of min over i of (inner i outside outer)."""
    cx, cy, S, oang = O
    m = len(P)
    best = -1e9
    for i in range(m):
        v = _hex_verts(P[i][0], P[i][1], A[i])
        for k in range(6):
            gap = _outer_sdf(v[k], cx, cy, S, oang)
            if gap > best:
                best = gap
    return best

def _objective(x, O):
    P = x[:2 * N].reshape(N, 2)
    A = x[2 * N:3 * N]
    return -min(_min_pair_gap(P, A), _min_inner_outer(P, A, O))

def _grad(x, O):
    g = np.zeros_like(x)
    h = 1e-5
    for k in range(len(x)):
        xp = x.copy(); xp[k] += h
        xm = x.copy(); xm[k] -= h
        g[k] = (_objective(xp, O) - _objective(xm, O)) / (2 * h)
    return g

def _eval(x, O):
    """Return (feasible, outer_side) for a configuration."""
    P = x[:2 * N].reshape(N, 2)
    A = x[2 * N:3 * N]
    cx, cy, S, oang = O
    # pair collision (SAT)
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap(_hex_verts(P[i][0], P[i][1], A[i]),
                            _hex_verts(P[j][0], P[j][1], A[j])):
                return False, None
    # inner inside outer
    for i in range(N):
        v = _hex_verts(P[i][0], P[i][1], A[i])
        for k in range(6):
            if _outer_sdf(v[k], cx, cy, S, oang) < -1e-7:
                return False, None
    return True, S

def _sat_overlap(p1, p2):
    """SAT: True if two convex polygons overlap (interiors)."""
    for poly in (p1, p2):
        n = len(poly)
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            nx, ny = -(by - ay), (bx - ax)
            L = math.hypot(nx, ny)
            if L == 0:
                continue
            nx /= L; ny /= L
            min1, max1 = 1e9, -1e9
            min2, max2 = 1e9, -1e9
            for px, py in p1:
                d = px * nx + py * ny
                if d < min1: min1 = d
                if d > max1: max1 = d
            for px, py in p2:
                d = px * nx + py * ny
                if d < min2: min2 = d
                if d > max2: max2 = d
            if max1 < min2 or max2 < min1:
                return False
    return True

def _cluster_seed():
    """Compact 2x3 hex lattice cluster (centers 2 apart, no overlap)."""
    pts = []
    for row in range(2):
        for col in range(3):
            x = 2.0 * col
            y = math.sqrt(3.0) * row
            pts.append([x - 2.0, y - math.sqrt(3.0) / 2.0])  # center cluster
    return pts

def _optimize(seed_pts, outer, budget):
    """Run L-BFGS-B from seed with given outer; return best feasible config or None."""
    P0 = np.array(seed_pts, dtype=float)
    A0 = np.zeros(N)
    x0 = np.concatenate([P0.ravel(), A0])
    # initial center/angle of cluster
    cx0 = float(np.mean(P0[:, 0]))
    cy0 = float(np.mean(P0[:, 1]))
    # start with a generous outer, then shrink
    S0 = outer
    O0 = (cx0, cy0, S0, 0.0)
    # ensure feasible start by expanding if needed
    S = S0
    for _ in range(40):
        feasible, _ = _eval(x0, (cx0, cy0, S, 0.0))
        if feasible:
            break
        S += 0.2
    O = (cx0, cy0, S, 0.0)
    best = None
    best_S = S
    t_end = time.time() + budget
    # iterative outer shrink
    for it in range(200):
        if time.time() > t_end - 2.0:
            break
        res = minimize(_objective, x0, args=(O,), method='L-BFGS-B',
                       jac=_grad, options={'maxiter': 30, 'maxfun': 400,
                                           'gtol': 1e-6, 'ftol': 1e-9})
        x1 = res.x
        # try to shrink outer side
        for shrink in range(60):
            if time.time() > t_end - 2.0:
                break
            S_new = max(1.5, best_S - 0.03)
            O_new = (cx0, cy0, S_new, 0.0)
            feas, _ = _eval(x1, O_new)
            if feas:
                best_S = S_new
                best = (x1, O_new)
            else:
                break
        x0 = x1
    return best

def construct():
    global _T0
    t0 = time.time()
    seed_pts = _cluster_seed()
    # initial outer estimate from bounding the cluster
    xs = [p[0] for p in seed_pts]; ys = [p[1] for p in seed_pts]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    outer_est = span + 3.0

    best = None
    best_S = 1e9
    # Phase 1: primary optimization
    budget1 = min(60.0, DEADLINE - (time.time() - t0) - 5.0)
    if budget1 > 10.0:
        r = _optimize(seed_pts, outer_est, budget1)
        if r is not None:
            x, O = r
            feas, S = _eval(x, O)
            if feas and S < best_S:
                best_S = S
                best = (x, O)

    # Phase 2: try a few different outer rotations / seed perturbations with remaining time
    if best is not None:
        xbest, Obest = best
        cx0 = Obest[0]; cy0 = Obest[1]
        for oang in (30.0, 60.0, 15.0):
            if time.time() > DEADLINE - 8.0:
                break
            S_try = max(1.5, best_S - 0.05)
            O_try = (cx0, cy0, S_try, oang)
            # quick local refine
            res = minimize(_objective, xbest, args=(O_try,), method='L-BFGS-B',
                           jac=_grad, options={'maxiter': 25, 'maxfun': 300, 'ftol': 1e-9})
            x1 = res.x
            for shrink in range(40):
                if time.time() > DEADLINE - 8.0:
                    break
                S_new = max(1.5, best_S - 0.03)
                O_new = (cx0, cy0, S_new, oang)
                feas, _ = _eval(x1, O_new)
                if feas:
                    best_S = S_new
                    best = (x1, O_new)
                else:
                    break

    if best is None:
        # fallback: trivial valid placement (large outer)
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    x, O = best
    cx, cy, S, oang = O
    P = x[:2 * N].reshape(N, 2)
    A = x[2 * N:3 * N]
    inner = [[float(P[i][0]), float(P[i][1]), float(A[i])] for i in range(N)]
    # final safety: verify, else expand
    for _ in range(50):
        if time.time() > DEADLINE - 2.0:
            break
        feas, _ = _eval(x, (cx, cy, S, oang))
        if feas:
            break
        S += 0.2
    return inner, (float(cx), float(cy)), float(S), float(oang)
