"""Circle packing in perimeter-4 rectangle: epigraph smooth objective + analytic
gradient + SLSQP + gradient flow + full polish ladder (free aspect ratio)."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
MARGIN = 1e-6          # safety margin for bounding-box perimeter
EPS = 1e-12


# ---------------------------------------------------------------------------
# Validity (bounding-box model: rectangle may shrink to circles' tight bbox)
# ---------------------------------------------------------------------------
def _is_valid(circles, tol=1e-9):
    if len(circles) != N:
        return False
    if np.any(circles[:, 2] <= 0):
        return False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    W = (np.max(xs + rs) - np.min(xs - rs))
    H = (np.max(ys + rs) - np.min(ys - rs))
    if W + H > 2.0 + tol:
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx * dx + dy * dy < (rs[i] + rs[j]) ** 2 - tol:
                return False
    return True


def _is_valid_fixed(x, y, r, W, H, tol=1e-9):
    """Validity against a fixed rectangle (used by the polish ladder)."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(x - r < -tol) or np.any(y - r < -tol):
        return False
    if np.any(x + r > W + tol) or np.any(y + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _max_radius_for_circle(x, y, r, w, h, i):
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j:
            continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r[j]
            if r_allowed < r_max:
                r_max = r_allowed
    return max(0.0, r_max)


# ---------------------------------------------------------------------------
# Smooth epigraph energy  E = -sum(r) + W*softmax(viol) + B*softmax(bviol)
# with auxiliary variable t (epigraph) -> always smooth, analytic gradient.
# ---------------------------------------------------------------------------
def _epigraph_energy(z, beta=30.0, Wv=500.0, Bv=500.0):
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w
    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
    viol = r_sum[iu] - d_pairs

    lv = r - x
    rv = x + r - w
    bv = r - y
    tv = y + r - h
    bviol = np.concatenate([lv, rv, bv, tv])

    e = -np.sum(r)
    # pairwise soft-max epigraph: max(viol, t)
    a = beta * (viol - t)
    a = np.clip(a, -500, 500)
    am = np.max(a)
    lse = am + np.log(np.sum(np.exp(a - am)))
    e += Wv * (t + lse / beta)
    # boundary soft-max epigraph
    b = beta * (bviol - t)
    b = np.clip(b, -500, 500)
    bm = np.max(b)
    lse_b = bm + np.log(np.sum(np.exp(b - bm)))
    e += Bv * (t + lse_b / beta)
    # keep aspect ratio in a sane range
    if w < 0.45:
        e += 200.0 * (0.45 - w) ** 2
    elif w > 1.55:
        e += 200.0 * (w - 1.55) ** 2
    return e


def _epigraph_grad(z, beta=30.0, Wv=500.0, Bv=500.0):
    x = z[0::3][:N].copy()
    y = z[1::3][:N].copy()
    r = z[2::3][:N].copy()
    w = z[-1]
    h = 2.0 - w
    if w < 0.2 or h < 0.2:
        return None

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
    viol = r_sum[iu] - d_pairs

    lv = r - x
    rv = x + r - w
    bv = r - y
    tv = y + r - h
    bviol = np.concatenate([lv, rv, bv, tv])

    # softmax weights for pairwise
    a = beta * (viol - t) if False else None  # t not needed; use softmax of viol
    a = beta * viol
    a = np.clip(a, -500, 500)
    am = np.max(a)
    sw = np.exp(a - am)
    sw /= (np.sum(sw) + 1e-30)

    b = beta * bviol
    b = np.clip(b, -500, 500)
    bm = np.max(b)
    swb = np.exp(b - bm)
    swb /= (np.sum(swb) + 1e-30)

    gx = np.zeros(N)
    gy = np.zeros(N)
    gr = np.full(N, -1.0)
    gw = 0.0

    for idx in range(len(iu[0])):
        i = iu[0][idx]
        j = iu[1][idx]
        wt = Wv * sw[idx]
        if wt < 1e-12:
            continue
        d = d_pairs[idx]
        if d < 1e-9:
            continue
        ux = (x[i] - x[j]) / d
        uy = (y[i] - y[j]) / d
        gx[i] += wt * ux
        gx[j] -= wt * ux
        gy[i] += wt * uy
        gy[j] -= wt * uy
        gr[i] += wt
        gr[j] += wt

    lw = swb[:N]
    rw = swb[N:2 * N]
    bw = swb[2 * N:3 * N]
    tw = swb[3 * N:4 * N]
    gx -= Bv * lw
    gx += Bv * rw
    gy -= Bv * bw
    gy += Bv * tw
    gr += Bv * (lw + rw + bw + tw)
    gw += Bv * (rw + tw)

    if w < 0.45:
        gw += 400.0 * (w - 0.45)
    elif w > 1.55:
        gw += 400.0 * (w - 1.55)

    g = np.empty_like(z)
    g[0::3][:N] = gx
    g[1::3][:N] = gy
    g[2::3][:N] = gr
    g[-1] = gw
    return g


# ---------------------------------------------------------------------------
# Force-directed relaxation (vectorized)
# ---------------------------------------------------------------------------
def _force_relax_unified(z, max_iter=500):
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 1e-4
    conv = 1e-8
    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w
        if w < 0.1 or h < 0.1:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)
        fx -= np.maximum(r - x, 0.0) * 100
        fx += np.maximum(x + r - w, 0.0) * 100
        fy -= np.maximum(r - y, 0.0) * 100
        fy += np.maximum(y + r - h, 0.0) * 100

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
        max_ov = float(np.max(overlap)) if overlap.size else 0.0
        if max_ov > conv:
            fp = overlap * 15.0
            dpx = dx[iu]
            dpy = dy[iu]
            ux = dpx / d_pairs
            uy = dpy / d_pairs
            for k in range(len(iu[0])):
                i = iu[0][k]
                j = iu[1][k]
                fx[i] += ux[k] * fp[k]
                fy[i] += uy[k] * fp[k]
                fx[j] -= ux[k] * fp[k]
                fy[j] -= uy[k] * fp[k]
        if max_ov < conv:
            break
        fm = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if fm > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif fm < 0.1:
            dt = min(0.01, dt * 1.1)
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])
    return z


# ---------------------------------------------------------------------------
# Gradient-descent flow on the epigraph energy (uses analytic gradient)
# ---------------------------------------------------------------------------
def _grad_flow(z, time_budget=12.0, beta=30.0):
    z = z.copy().astype(float)
    start = time.time()
    lr = 0.003
    best_e = _epigraph_energy(z, beta)
    best_z = z.copy()
    patience = 0
    for it in range(4000):
        if time.time() - start > time_budget:
            break
        g = _epigraph_grad(z, beta)
        if g is None:
            break
        z_new = z - lr * g
        for i in range(N):
            w = z_new[-1]
            h = 2.0 - w
            r_i = max(z_new[3 * i + 2], 1e-4)
            z_new[3 * i] = np.clip(z_new[3 * i], r_i, max(w - r_i, r_i))
            z_new[3 * i + 1] = np.clip(z_new[3 * i + 1], r_i, max(h - r_i, r_i))
            z_new[3 * i + 2] = np.clip(z_new[3 * i + 2], 1e-4, 0.5)
        z_new[-1] = np.clip(z_new[-1], 0.3, 1.7)
        e = _epigraph_energy(z_new, beta)
        if e < best_e - 1e-9:
            best_e = e
            best_z = z_new.copy()
            patience = 0
            lr = min(lr * 1.05, 0.012)
        else:
            patience += 1
            lr *= 0.95
            if patience > 60:
                break
    return best_z


# ---------------------------------------------------------------------------
# Polish ladder (kept from the current program, using fixed-rectangle validity)
# ---------------------------------------------------------------------------
def _greedy_expand(x, y, r, w, h, time_budget):
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    improved = True
    rounds = 0
    while improved and rounds < 50:
        if time.time() > time_budget:
            break
        improved = False
        rounds += 1
        for i in range(N):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-8:
                r[i] = r_max
                improved = True
    return x, y, r


def _polish_result(z, time_budget):
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    dt = 0.001
    max_iter = int(min(3000, time_budget * 150))
    for it in range(max_iter):
        if time.time() > time_budget:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)
        fx -= np.maximum(r - x, 0.0) * 35
        fx += np.maximum(x + r - w, 0.0) * 35
        fy -= np.maximum(r - y, 0.0) * 35
        fy += np.maximum(y + r - h, 0.0) * 35
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
        if np.max(overlap) > 1e-10:
            fp = overlap * 18.0
            dpx = dx[iu]
            dpy = dy[iu]
            ux = dpx / d_pairs
            uy = dpy / d_pairs
            for k in range(len(iu[0])):
                i = iu[0][k]
                j = iu[1][k]
                fx[i] += ux[k] * fp[k]
                fy[i] += uy[k] * fp[k]
                fx[j] -= ux[k] * fp[k]
                fy[j] -= uy[k] * fp[k]
        mf = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        dt = 0.0005 if mf > 0.1 else 0.001
        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
    gf = 1.0003
    for _ in range(300):
        if time.time() > time_budget:
            break
        rt = r * gf
        if _is_valid_fixed(x, y, rt, w, h):
            r = rt
        else:
            break
    zn = np.empty(3 * N + 1)
    zn[0::3][:n] = x
    zn[1::3][:n] = y
    zn[2::3][:n] = r
    zn[-1] = w
    return zn if _is_valid_fixed(x, y, r, w, h) else None


def _is_valid_move(x, y, r, w, h, i, nx, ny):
    if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
        return False
    for j in range(N):
        if i == j:
            continue
        ddx = nx - x[j]
        ddy = ny - y[j]
        rs = r[i] + r[j]
        if ddx * ddx + ddy * ddy < rs * rs - 1e-9:
            return False
    return True


def _expansion_relaxation_polish(z, time_budget):
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    for rnd in range(30):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                ra = d - r[j]
                if ra < r_max:
                    r_max = ra
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-6:
                lo, hi = r[i], r_max
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    rr = r.copy()
                    rr[i] = mid
                    if _is_valid_move(x, y, rr, w, h, i, x[i], y[i]):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-7:
                    r[i] = lo
                    improved = True
            mind = float('inf')
            blk = None
            dl, dr, db, dt_ = x[i] - r[i], w - x[i] - r[i], y[i] - r[i], h - y[i] - r[i]
            if dl < mind:
                mind, blk = dl, 'left'
            if dr < mind:
                mind, blk = dr, 'right'
            if db < mind:
                mind, blk = db, 'bottom'
            if dt_ < mind:
                mind, blk = dt_, 'top'
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < mind:
                    mind, blk = gap, j
            if mind < 0.02:
                mx = my = 0.0
                if blk == 'left':
                    mx = 1.0
                elif blk == 'right':
                    mx = -1.0
                elif blk == 'bottom':
                    my = 1.0
                elif blk == 'top':
                    my = -1.0
                else:
                    j = blk
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d > 1e-6:
                        mx, my = dx / d, dy / d
                for step in [0.05, 0.03, 0.02, 0.01, 0.005, 0.002]:
                    if _is_valid_move(x, y, r, w, h, i, x[i] + mx * step, y[i] + my * step):
                        x[i] += mx * step
                        y[i] += my * step
                        improved = True
                        break
                else:
                    px, py = -my, mx
                    for step in [0.03, 0.01, 0.005]:
                        for s in (1.0, -1.0):
                            if _is_valid_move(x, y, r, w, h, i, x[i] + s * px * step, y[i] + s * py * step):
                                x[i] += s * px * step
                                y[i] += s * py * step
                                improved = True
                                break
                        if improved:
                            break
        if not improved:
            break
    gf = 1.001
    for _ in range(100):
        if time.time() > time_budget:
            break
        rt = r * gf
        if _is_valid_fixed(x, y, rt, w, h):
            r = rt
        else:
            break
    if _is_valid_fixed(x, y, r, w, h):
        zn = np.empty(3 * N + 1)
        zn[0::3][:n] = x
        zn[1::3][:n] = y
        zn[2::3][:n] = r
        zn[-1] = w
        return zn
    return None


def _local_search_positions(x, y, r, w, h, time_budget):
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    rng = np.random.default_rng(123)
    best = np.sum(r)
    for _ in range(5000):
        if time.time() > time_budget:
            break
        i = rng.integers(0, N)
        nx = x[i] + rng.normal(0, 0.005)
        ny = y[i] + rng.normal(0, 0.005)
        if not _is_valid_move(x, y, r, w, h, i, nx, ny):
            continue
        x[i] = nx
        y[i] = ny
        x2, y2, r2 = _greedy_expand(x, y, r, w, h, time_budget)
        if np.sum(r2) > best:
            best = np.sum(r2)
            r, x, y = r2, x2, y2
        else:
            x[i] -= nx - x[i]
            y[i] -= ny - y[i]
            r = r.copy()
    return x, y, r


def _constraint_refine(x, y, r, w, h, time_budget):
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    n = N
    for rnd in range(500):
        if time.time() > time_budget:
            break
        changed = False
        for _ in range(10):
            for i in range(n):
                rm = _max_radius_for_circle(x, y, r, w, h, i)
                if rm > r[i] + 1e-10:
                    r[i] = rm
                    changed = True
            if not changed:
                break
        if not changed:
            break
        for i in range(n):
            rm = _max_radius_for_circle(x, y, r, w, h, i)
            if rm <= r[i] + 1e-8:
                continue
            mingap = float('inf')
            bdx = bdy = 0.0
            for gap, dx, dy in [(x[i] - r[i], 1, 0), (w - x[i] - r[i], -1, 0),
                                (y[i] - r[i], 0, 1), (h - y[i] - r[i], 0, -1)]:
                if gap < mingap:
                    mingap, bdx, bdy = gap, dx, dy
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-12:
                    continue
                gap = d - (r[i] + r[j])
                if gap < mingap:
                    mingap, bdx, bdy = gap, dx / d, dy / d
            for step in [0.002, 0.001, 0.0005, 0.0002]:
                nx, ny = x[i] + bdx * step, y[i] + bdy * step
                if _is_valid_move(x, y, r, w, h, i, nx, ny):
                    x[i], y[i] = nx, ny
                    break
    return x, y, r


def _high_precision_polish(x, y, r, w, h, time_budget):
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    n = N
    tol = 1e-12
    for rnd in range(200):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            rm = _max_radius_for_circle(x, y, r, w, h, i)
            if rm > r[i] + tol:
                r[i] = rm
                improved = True
        if improved:
            continue
        for i in range(n):
            rmb = _max_radius_for_circle(x, y, r, w, h, i)
            if rmb <= r[i] + tol:
                continue
            for dx in [0.001, -0.001, 0.0005, -0.0005, 0.0]:
                for dy in [0.001, -0.001, 0.0005, -0.0005, 0.0]:
                    if dx == 0.0 and dy == 0.0:
                        continue
                    nx, ny = x[i] + dx, y[i] + dy
                    if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
                        continue
                    ok = True
                    for j in range(n):
                        if i == j:
                            continue
                        ddx, ddy = nx - x[j], ny - y[j]
                        rs = r[i] + r[j]
                        if ddx * ddx + ddy * ddy < rs * rs - tol:
                            ok = False
                            break
                    if ok:
                        x[i], y[i] = nx, ny
                        rnm = _max_radius_for_circle(x, y, r, w, h, i)
                        if rnm > r[i] + tol:
                            r[i] = rnm
                            improved = True
                        break
        if not improved:
            for factor in [1.0 + 1e-6, 1.0 + 5e-7, 1.0 + 1e-7, 1.0 + 5e-8]:
                rt = r * factor
                if _is_valid_fixed(x, y, rt, w, h, tol=1e-10):
                    r = rt
                    improved = True
                    break
        if not improved:
            break
    return x, y, r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()
    deadline = start_time + 138.0
    beta = 30.0
    best_value = 0.0
    best_result = None

    def check(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        if np.any(r <= 0):
            return False, 0.0
        W = (np.max(x + r) - np.min(x - r))
        H = (np.max(y + r) - np.min(y - r))
        if W + H > 2.0 - MARGIN:
            return False, 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                    return False, 0.0
        return True, float(np.sum(r))

    def consider(z):
        nonlocal best_value, best_result
        if z is None:
            return
        ok, val = check(z)
        if ok and val > best_value:
            best_value = val
            best_result = z.copy()

    # ---- initial guesses (fixed-rectangle layout, free aspect ratio) ----
    def get_guesses():
        gs = []
        side = int(np.ceil(np.sqrt(N)))
        for w0 in [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 1.2, 0.8]:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                z[3 * i] = (col + 0.5) / side * w0
                z[3 * i + 1] = (row + 0.5) / side * h0
                z[3 * i + 2] = min(w0 / side, h0 / side) * 0.45
            z[-1] = w0
            gs.append(z)
        rng = np.random.default_rng(42)
        for _ in range(12):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                z[3 * i] = rng.uniform(r0, w0 - r0)
                z[3 * i + 1] = rng.uniform(r0, h0 - r0)
                z[3 * i + 2] = r0
            z[-1] = w0
            gs.append(z)
        # hexagonal
        w0 = h0 = 1.0
        z = np.zeros(3 * N + 1)
        cols = 5
        sx, sy = w0 / (cols + 0.5), h0 / (5 + 0.5)
        for i in range(N):
            row, col = divmod(i, cols)
            cx = (col + 0.5 + (0.5 if row % 2 == 1 else 0)) * sx
            cy = (row + 0.5) * sy
            r0 = min(sx, sy) * 0.48
            z[3 * i] = min(max(cx, r0), w0 - r0)
            z[3 * i + 1] = min(max(cy, r0), h0 - r0)
            z[3 * i + 2] = r0
        z[-1] = w0
        gs.append(z)
        return gs

    guesses = get_guesses()
    bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]

    # ---- Phase 1: SLSQP (epigraph objective + analytic gradient) ----
    for z0 in guesses:
        if time.time() > deadline - 35:
            break
        z0 = _force_relax_unified(z0, max_iter=300)
        remaining = deadline - (time.time() - start_time) - 35
        if remaining < 5:
            break
        maxiter = int(min(400, max(80, remaining * 30)))
        try:
            res = minimize(_epigraph_energy, z0, jac=_epigraph_grad, method='SLSQP',
                           bounds=bounds,
                           args=(beta,),
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 60})
            consider(res.x)
            # repair
            zr = res.x.copy()
            zr[2::3][:N] *= 0.95
            consider(zr)
        except Exception:
            continue

    # ---- Phase 2: gradient-descent flow on best + diverse starts ----
    if time.time() < deadline - 30:
        flow_starts = []
        if best_result is not None:
            flow_starts.append(best_result.copy())
        for z0 in guesses[:3]:
            flow_starts.append(_force_relax_unified(z0, max_iter=200))
        for z0 in flow_starts:
            if time.time() > deadline - 30:
                break
            remaining = deadline - (time.time() - start_time) - 30
            if remaining < 5:
                break
            zf = _grad_flow(z0, time_budget=min(remaining - 2, 12.0), beta=beta)
            consider(zf)
            # SLSQP polish from flow result
            try:
                res = minimize(_epigraph_energy, zf, jac=_epigraph_grad, method='SLSQP',
                               bounds=bounds, args=(beta,),
                               options={'maxiter': 200, 'ftol': 1e-12, 'maxls': 60})
                consider(res.x)
            except Exception:
                pass

    # ---- Phase 3: polish ladder (fixed-rectangle) ----
    if best_result is not None and time.time() < deadline - 20:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w

        def to_z(x, y, r, w):
            zn = np.empty(3 * N + 1)
            zn[0::3][:N] = x
            zn[1::3][:N] = y
            zn[2::3][:N] = r
            zn[-1] = w
            return zn

        x, y, r = _greedy_expand(x, y, r, w, h, deadline - 15)
        if _is_valid_fixed(x, y, r, w, h):
            consider(to_z(x, y, r, w))

        if time.time() < deadline - 15:
            consider(_polish_result(best_result, deadline - (time.time() - start_time) - 10))

        if time.time() < deadline - 10:
            consider(_expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 5))

        if time.time() < deadline - 8:
            x, y, r = _local_search_positions(x, y, r, w, h, deadline - 4)
            if _is_valid_fixed(x, y, r, w, h):
                consider(to_z(x, y, r, w))

        if time.time() < deadline - 6:
            x, y, r = _constraint_refine(x, y, r, w, h, deadline - 3)
            if _is_valid_fixed(x, y, r, w, h):
                consider(to_z(x, y, r, w))

        if time.time() < deadline - 2:
            x, y, r = _high_precision_polish(x, y, r, w, h, deadline - 1)
            if _is_valid_fixed(x, y, r, w, h):
                consider(to_z(x, y, r, w))

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w
    for _ in range(20):
        rt = r * (1 - 1e-10)
        if _is_valid_fixed(x, y, rt, w, h, tol=1e-9):
            r = rt
        else:
            break
    return np.column_stack((x, y, r))
