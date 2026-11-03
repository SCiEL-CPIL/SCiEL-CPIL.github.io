import math
import time

N = 12
T0 = time.time()
DEADLINE = T0 + 130.0

# ---------- geometry ----------
def hex_verts(cx, cy, s, a):
    a = math.radians(a)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for k in range(6):
        t = math.radians(60.0 * k + a)
        x = s * math.cos(t)
        y = s * math.sin(t)
        out.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return out

def proj_range(pts, ux, uy):
    mn = mx = None
    for (x, y) in pts:
        d = x * ux + y * uy
        if mn is None or d < mn:
            mn = d
        if mx is None or d > mx:
            mx = d
    return mn, mx

def sat_gap(A, B):
    """Gap between two convex polygons along a separating axis; >0 means disjoint.
    Returns (gap, ux, uy) with the maximum gap over all candidate axes."""
    best = -1.0
    bux = buy = 0.0
    for poly in (A, B):
        m = len(poly)
        for i in range(m):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % m]
            ex = x2 - x1
            ey = y2 - y1
            L = math.hypot(ex, ey)
            if L < 1e-12:
                continue
            ux = -ey / L
            uy = ex / L
            a0, a1 = proj_range(A, ux, uy)
            b0, b1 = proj_range(B, ux, uy)
            gap = min(a1, b1) - max(a0, b0)
            if gap > best:
                best = gap
                bux, buy = ux, uy
    return best, bux, buy

def fit_side(cx, cy, s, a):
    """Minimum outer side length for given center/rotation to contain all unit hexes."""
    best = 0.0
    for k in range(6):
        t = math.radians(60.0 * k + a)
        ux, uy = math.cos(t), math.sin(t)
        mx = -1e30
        for (px, py) in hex_verts(0.0, 0.0, 1.0, 0.0):
            d = (px - cx) * ux + (py - cy) * uy
            if d > mx:
                mx = d
        if mx > best:
            best = mx
    return best

# ---------- smooth objective ----------
EPS = 1.0
def softmin(vals):
    m = max(vals)
    z = math.exp(-EPS * (m - min(vals)))
    return m - math.log(1.0 + z) / EPS

def smooth_fit(x, y, a):
    """Differentiable (smooth) estimate of the outer side length."""
    ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
    best = 0.0
    for k in range(6):
        t = math.radians(60.0 * k + a)
        dx, dy = math.cos(t), math.sin(t)
        # rotate normal into world frame
        ux = ca * dx - sa * dy
        uy = sa * dx + ca * dy
        vals = []
        for (px, py) in hex_verts(0.0, 0.0, 1.0, 0.0):
            vals.append((px - x) * ux + (py - y) * uy)
        sm = softmin(vals)
        if sm > best:
            best = sm
    return best

def smooth_gap(A, B):
    g, _, _ = sat_gap(A, B)
    return -softmin([g, 0.0])

def objective_and_grads(v):
    """v = [x1,y1,a1,...,x12,y12,a12, cx,cy,s,a]. Returns (f, grad[39])."""
    n = N
    grads = [0.0] * (3 * n + 4)
    cx, cy, s, a = v[3 * n], v[3 * n + 1], v[3 * n + 2], v[3 * n + 3]
    ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))

    # outer support (smooth)
    best = 0.0
    bestk = 0
    for k in range(6):
        t = math.radians(60.0 * k + a)
        dx, dy = math.cos(t), math.sin(t)
        ux = ca * dx - sa * dy
        uy = sa * dx + ca * dy
        vals = []
        for (px, py) in hex_verts(0.0, 0.0, 1.0, 0.0):
            vals.append((px - cx) * ux + (py - cy) * uy)
        sm = softmin(vals)
        if sm > best:
            best = sm
            bestk = k
    f = best
    # gradient of active support term w.r.t. cx, cy, a
    t = math.radians(60.0 * bestk + a)
    dx, dy = math.cos(t), math.sin(t)
    ux = ca * dx - sa * dy
    uy = sa * dx + ca * dy
    # sum over hex vertices of d/dparam of ((p - c)·u)
    sx = sy = sa_sum = 0.0
    for (px, py) in hex_verts(0.0, 0.0, 1.0, 0.0):
        sx += ux
        sy += uy
        # d/d(a) of u·p where u rotates with a
        # du/da (radians) = (-dy, dx); contribution = (p - c)·du/da
        sa_sum += (px - cx) * (-dy) + (py - cy) * dx
    grads[3 * n] = -sx
    grads[3 * n + 1] = -sy
    grads[3 * n + 3] = sa_sum / 57.29577951308232
    # (no direct gradient for s in objective; s is a variable only for constraints)

    # pairwise non-overlap penalties
    verts = [hex_verts(v[3 * i], v[3 * i + 1], 1.0, v[3 * i + 2]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g, ux, uy = sat_gap(verts[i], verts[j])
            pen = -softmin([g, 0.0])
            f += pen
            # gradient of pen w.r.t. positions of i and j
            for (px, py) in verts[i]:
                grads[3 * i] += pen * ux
                grads[3 * i + 1] += pen * uy
            for (px, py) in verts[j]:
                grads[3 * j] -= pen * ux
                grads[3 * j + 1] -= pen * uy
    return f, grads

def constraints_and_grads(v):
    n = N
    c = []
    g = []
    # containment: each hex support <= s
    cx, cy, s, a = v[3 * n], v[3 * n + 1], v[3 * n + 2], v[3 * n + 3]
    ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
    for i in range(n):
        A = hex_verts(v[3 * i], v[3 * i + 1], 1.0, v[3 * i + 2])
        for k in range(6):
            t = math.radians(60.0 * k + a)
            dx, dy = math.cos(t), math.sin(t)
            ux = ca * dx - sa * dy
            uy = sa * dx + ca * dy
            m = max((px - cx) * ux + (py - cy) * uy for (px, py) in A)
            c.append(m - s)
            gi = [0.0] * (3 * n + 4)
            for idx, (px, py) in enumerate(A):
                d = (px - cx) * ux + (py - cy) * uy
                if d == m:
                    gi[3 * i] += ux
                    gi[3 * i + 1] += uy
            gi[3 * n] = -m * 0.0  # cx term: derivative of m wrt cx = -ux for active
            # recompute properly:
            gi[3 * n] = 0.0
            gi[3 * n + 1] = 0.0
            for (px, py) in A:
                d = (px - cx) * ux + (py - cy) * uy
                if d == m:
                    gi[3 * n] += -ux
                    gi[3 * n + 1] += -uy
            gi[3 * n + 2] = -1.0
            g.append(gi)
    # non-overlap: gap >= 0
    verts = [hex_verts(v[3 * i], v[3 * i + 1], 1.0, v[3 * i + 2]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            gg, ux, uy = sat_gap(verts[i], verts[j])
            c.append(gg)
            gi = [0.0] * (3 * n + 4)
            for (px, py) in verts[i]:
                gi[3 * i] += ux
                gi[3 * i + 1] += uy
            for (px, py) in verts[j]:
                gi[3 * j] -= ux
                gi[3 * j + 1] -= uy
            g.append(gi)
    return c, g

def _cons(v):
    c, _ = constraints_and_grads(v)
    return c

def _consJ(v):
    c, g = constraints_and_grads(v)
    return g

def _obj(v):
    f, _ = objective_and_grads(v)
    return f

def _objG(v):
    f, g = objective_and_grads(v)
    return g

def run_slsqp(v0, maxiter=40):
    from scipy.optimize import minimize
    n = N
    bounds = [(None, None)] * (3 * n) + [(None, None), (None, None), (1.5, 20.0), (-180.0, 180.0)]
    res = minimize(
        _obj, v0, jac=_objG, method='SLSQP', bounds=bounds,
        constraints=[{'type': 'ineq', 'fun': _cons, 'jac': _consJ}],
        options={'maxiter': maxiter, 'ftol': 1e-9, 'disp': False},
    )
    return res

def polish(v):
    """Refine outer center/rotation for the fixed inner configuration."""
    n = N
    cx, cy, s, a = v[3 * n], v[3 * n + 1], v[3 * n + 2], v[3 * n + 3]
    best = fit_side(cx, cy, s, a)
    best_c = (cx, cy, a)
    # local sweep over rotation
    for da in [0, 5, 10, 15, 30, -5, -10, -15, -30]:
        aa = a + da
        b = fit_side(cx, cy, s, aa)
        if b < best:
            best = b
            best_c = (cx, cy, aa)
    cx, cy, a = best_c
    # local sweep over center
    for dx in [0, 0.2, -0.2, 0.4, -0.4]:
        for dy in [0, 0.2, -0.2, 0.4, -0.4]:
            b = fit_side(cx + dx, cy + dy, s, a)
            if b < best:
                best = b
                cx, cy = cx + dx, cy + dy
    return best, (cx, cy, a)

def valid(v):
    n = N
    cx, cy, s, a = v[3 * n], v[3 * n + 1], v[3 * n + 2], v[3 * n + 3]
    # containment
    for i in range(n):
        A = hex_verts(v[3 * i], v[3 * i + 1], 1.0, v[3 * i + 2])
        for k in range(6):
            t = math.radians(60.0 * k + a)
            dx, dy = math.cos(t), math.sin(t)
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            ux = ca * dx - sa * dy
            uy = sa * dx + ca * dy
            m = max((px - cx) * ux + (py - cy) * uy for (px, py) in A)
            if m > s + 1e-6:
                return False
    # non-overlap
    verts = [hex_verts(v[3 * i], v[3 * i + 1], 1.0, v[3 * i + 2]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g, _, _ = sat_gap(verts[i], verts[j])
            if g < -1e-6:
                return False
    return True

def to_result(v):
    n = N
    inner = [[v[3 * i], v[3 * i + 1], v[3 * i + 2]] for i in range(n)]
    cx, cy, s, a = v[3 * n], v[3 * n + 1], v[3 * n + 2], v[3 * n + 3]
    return inner, (cx, cy), s, a

# ---------- initial layouts ----------
def layout_honeycomb():
    # hex grid: axial coords, pointy-top, spacing 2
    v = [0.0] * (3 * N + 4)
    # pick 12 compact axial positions
    ax = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
          (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2)]
    for i, (q, r) in enumerate(ax):
        x = 2.0 * (q + r / 2.0)
        y = math.sqrt(3.0) * r
        v[3 * i] = x
        v[3 * i + 1] = y
        v[3 * i + 2] = 0.0
    v[3 * N + 2] = 6.0
    return v

def layout_circular():
    v = [0.0] * (3 * N + 4)
    R = 3.5
    for i in range(N):
        t = 2.0 * math.pi * i / N
        v[3 * i] = R * math.cos(t)
        v[3 * i + 1] = R * math.sin(t)
        v[3 * i + 2] = 0.0
    v[3 * N + 2] = 8.0
    return v

def layout_ring2():
    v = [0.0] * (3 * N + 4)
    # 6 inner ring + 6 outer ring
    for i in range(6):
        t = 2.0 * math.pi * i / 6
        v[3 * i] = 2.2 * math.cos(t)
        v[3 * i + 1] = 2.2 * math.sin(t)
        v[3 * i + 2] = 0.0
    for i in range(6):
        t = 2.0 * math.pi * (i + 0.5) / 6
        v[3 * (i + 6)] = 4.6 * math.cos(t)
        v[3 * (i + 6) + 1] = 4.6 * math.sin(t)
        v[3 * (i + 6) + 2] = 0.0
    v[3 * N + 2] = 9.0
    return v

def construct():
    best_v = None
    best_s = 1e9

    candidates = [layout_honeycomb(), layout_circular(), layout_ring2()]
    # also a trivial spread as fallback
    v0 = [0.0] * (3 * N + 4)
    for i in range(N):
        v0[3 * i] = 3.0 * i
        v0[3 * i + 2] = 0.0
    v0[3 * N + 2] = 40.0
    candidates.append(v0)

    for v in candidates:
        if time.time() > DEADLINE:
            break
        # polish initial outer to get a feasible-ish starting point
        s, (cx, cy, a) = polish(v)
        v[3 * N] = cx
        v[3 * N + 1] = cy
        v[3 * N + 2] = s
        v[3 * N + 3] = a
        try:
            res = run_slsqp(v, maxiter=40)
            if res.success or (res.fun < 1e9):
                v2 = list(res.x)
                s2, (cx2, cy2, a2) = polish(v2)
                v2[3 * N] = cx2
                v2[3 * N + 1] = cy2
                v2[3 * N + 2] = s2
                v2[3 * N + 3] = a2
                if valid(v2):
                    if s2 < best_s:
                        best_s = s2
                        best_v = v2
        except Exception:
            pass
        # also accept the polished initial if valid
        if valid(v):
            if s < best_s:
                best_s = s
                best_v = v

    if best_v is None:
        # fallback: return the trivial seed-like valid layout
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # final polish
    s, (cx, cy, a) = polish(best_v)
    inner = [[best_v[3 * i], best_v[3 * i + 1], best_v[3 * i + 2]] for i in range(N)]
    return inner, (cx, cy), s, a
