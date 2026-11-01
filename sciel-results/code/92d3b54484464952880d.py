import numpy as np
from scipy.optimize import minimize
import time

N = 32
TIME_LIMIT = 142.0  # evaluator kills at 150s; keep margin

def _elapsed(start):
    return time.time() - start

def _is_valid(x, y, r, eps=1e-7):
    if np.any(x < -eps) or np.any(x > 1.0 + eps):
        return False
    if np.any(y < -eps) or np.any(y > 1.0 + eps):
        return False
    if np.any(r < -eps):
        return False
    if np.any(x < r - eps) or np.any(x > 1.0 - r + eps):
        return False
    if np.any(y < r - eps) or np.any(y > 1.0 - r + eps):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < (r[i] + r[j]) - eps:
                return False
    return True

def _sum_r(r):
    return float(np.sum(r))

def _objective(params, start):
    if _elapsed(start) > TIME_LIMIT:
        return 1e10
    x = params[:N]
    y = params[N:2*N]
    r = np.maximum(params[2*N:3*N], 0.0)
    obj = -np.sum(r)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + 1e-8))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])
    def sm(v):
        return 0.5 * (v + np.sqrt(v*v + 1e-8))
    bound_pen = (np.sum(sm(r - x)) + np.sum(sm(x - (1.0 - r)))
                 + np.sum(sm(r - y)) + np.sum(sm(y - (1.0 - r))))
    mu = 50.0
    return obj + mu * (overlap_pen + bound_pen)

def _objective_tight(params, start):
    if _elapsed(start) > TIME_LIMIT:
        return 1e10
    x = params[:N]
    y = params[N:2*N]
    r = np.maximum(params[2*N:3*N], 0.0)
    obj = -np.sum(r)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + 1e-12))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])
    def sm(v):
        return 0.5 * (v + np.sqrt(v*v + 1e-12))
    bound_pen = (np.sum(sm(r - x)) + np.sum(sm(x - (1.0 - r)))
                 + np.sum(sm(r - y)) + np.sum(sm(y - (1.0 - r))))
    mu = 200.0
    return obj + mu * (overlap_pen + bound_pen)

def _objective_grad(params, start, mu=200.0, eps=1e-12):
    """Tight objective WITH analytic gradient for L-BFGS-B."""
    if _elapsed(start) > TIME_LIMIT:
        return 1e10, np.zeros_like(params)
    x = params[:N].copy()
    y = params[N:2*N].copy()
    r = np.maximum(params[2*N:3*N], 0.0)
    obj = -np.sum(r)
    gx = np.zeros(N); gy = np.zeros(N); gr = np.zeros(N)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    s = np.sqrt(diff*diff + eps*eps)
    w = np.where(mask, 0.5*(1.0 + diff/s), 0.0)   # weight for each pair
    # gradient of smooth_max wrt dist: -w ; wrt r: +w
    # d(dist)/d x[i] = (x[i]-x[j])/dist for pair (i,j)
    for i in range(N):
        for j in range(i+1, N):
            wij = w[i, j]
            dij = dist[i, j]
            gx[i] += wij * dx[i, j] / dij
            gy[i] += wij * dy[i, j] / dij
            gx[j] -= wij * dx[i, j] / dij
            gy[j] -= wij * dy[i, j] / dij
            gr[i] += wij
            gr[j] += wij
    overlap_pen = np.sum(w[mask])

    def sm(v):
        return 0.5*(v + np.sqrt(v*v + eps*eps))
    def smg(v):
        return 0.5*(1.0 + v/np.sqrt(v*v + eps*eps))
    dneg_x = r - x; dpos_x = x - (1.0 - r)
    dneg_y = r - y; dpos_y = y - (1.0 - r)
    bound_pen = (np.sum(sm(dneg_x)) + np.sum(sm(dpos_x))
                 + np.sum(sm(dneg_y)) + np.sum(sm(dpos_y)))
    gx += smg(dneg_x)      # d(dneg_x)/dx = -1 -> -smg ; but dneg_x=r-x, d/dx=-1
    # careful: d/dx of sm(r-x) = smg(r-x)*(-1)
    gx -= smg(dneg_x)
    gx += smg(dpos_x)      # d/dx of sm(x-(1-r)) = smg
    gy -= smg(dneg_y)
    gy += smg(dpos_y)
    gr += smg(dneg_x) + smg(dneg_y)          # d/d r of sm(r-x)=smg, sm(r-y)=smg
    gr += smg(dpos_x)*(-1) + smg(dpos_y)*(-1)  # d/d r of sm(x-(1-r)) = smg*(+1)?? x-(1-r)=x-1+r, d/dr=+1

    total = obj + mu*(overlap_pen + bound_pen)
    grad = np.concatenate([gx, gy, gr]) * mu
    grad[2*N:3*N] -= 1.0  # d(-sum r)/dr = -1
    return total, grad

def _push_apart(x, y, r, iters=400):
    x = x.copy(); y = y.copy()
    for _ in range(iters):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy + 1e-14)
        need = r[:, None] + r[None, :]
        overlaps = dist < need
        if not np.any(overlaps):
            break
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                if not overlaps[i, j]:
                    continue
                d = dist[i, j]
                need_ij = r[i] + r[j]
                if d < need_ij:
                    if d < 1e-12:
                        ang = (i*0.7 + j*1.3) % (2*np.pi)
                        ux, uy = np.cos(ang), np.sin(ang)
                    else:
                        ux, uy = dx[i, j]/d, dy[i, j]/d
                    step = 0.5*(need_ij - d)
                    x[i] += step*ux; y[i] += step*uy
                    x[j] -= step*ux; y[j] -= step*uy
                    moved = True
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        if not moved:
            break
    return x, y

def _physics_optimize(x, y, r, start):
    """Repulsive-particle relaxation: grow radii, push centers apart, keep in bounds."""
    x = x.copy(); y = y.copy(); r = r.copy()
    best = (x.copy(), y.copy(), r.copy())
    best_score = _sum_r(r)
    # grow radii toward boundary limit
    for it in range(2000):
        if _elapsed(start) > TIME_LIMIT - 2:
            break
        # target radius for each circle = min distance to boundary
        tgt = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        r = np.minimum(np.maximum(r, 0.0), tgt)
        # repulsion: push apart overlapping pairs
        for _ in range(60):
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx*dx + dy*dy + 1e-14)
            need = r[:, None] + r[None, :]
            overlaps = dist < need
            if not np.any(overlaps):
                break
            for i in range(N):
                for j in range(i+1, N):
                    if not overlaps[i, j]:
                        continue
                    d = dist[i, j]
                    if d < need[i, j]:
                        if d < 1e-12:
                            ang = (i*0.7+j*1.3) % (2*np.pi)
                            ux, uy = np.cos(ang), np.sin(ang)
                        else:
                            ux, uy = dx[i, j]/d, dy[i, j]/d
                        step = 0.5*(need[i, j] - d)
                        x[i] += step*ux; y[i] += step*uy
                        x[j] -= step*ux; y[j] -= step*uy
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        # try to grow radii a bit more
        grow = 1.0 + 0.002
        r2 = np.minimum(r*grow, np.minimum(np.minimum(x, 1.0-x), np.minimum(y, 1.0-y)))
        # accept only if still separable
        ok = True
        for _ in range(40):
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx*dx + dy*dy + 1e-14)
            need = r2[:, None] + r2[None, :]
            if not np.any(dist < need):
                break
            for i in range(N):
                for j in range(i+1, N):
                    if dist[i, j] < need[i, j]:
                        d = dist[i, j]
                        if d < 1e-12:
                            ang = (i*0.7+j*1.3) % (2*np.pi)
                            ux, uy = np.cos(ang), np.sin(ang)
                        else:
                            ux, uy = dx[i, j]/d, dy[i, j]/d
                        step = 0.5*(need[i, j] - d)
                        x[i] += step*ux; y[i] += step*uy
                        x[j] -= step*ux; y[j] -= step*uy
        x = np.clip(x, r2, 1.0 - r2)
        y = np.clip(y, r2, 1.0 - r2)
        r = r2
        s = _sum_r(r)
        if s > best_score:
            best_score = s
            best = (x.copy(), y.copy(), r.copy())
    return best

def _make_valid(x, y, r):
    x, y = _push_apart(x, y, r)
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    for _ in range(300):
        ok = True
        for i in range(N):
            mr = min(min(x[i], 1.0-x[i]), min(y[i], 1.0-y[i]))
            if r[i] > mr:
                r[i] = max(0.0, mr*0.999)
                ok = False
        if not ok:
            continue
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy + 1e-14)
        need = r[:, None] + r[None, :]
        overlaps = dist < need
        if np.any(overlaps):
            idx = np.where(overlaps)
            i, j = idx[0][0], idx[1][0]
            if j < i:
                i, j = j, i
            r[i] *= 0.995
            r[j] *= 0.995
            ok = False
        if ok:
            break
    return x, y, r

def _hex_init():
    pts = []
    rowh = np.sqrt(3.0)/2.0
    r = 0
    while len(pts) < N:
        for c in range(0, 12):
            x = c + (0.5 if (r % 2) else 0.0)
            y = r * rowh
            pts.append([x, y])
            if len(pts) >= N:
                break
        r += 1
    pts = np.array(pts[:N])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin)
    if span < 1e-9:
        span = 1.0
    scale = 0.85 / span
    x = (pts[:,0] - xmin) * scale + 0.075
    y = (pts[:,1] - ymin) * scale + 0.075
    r = np.zeros(N)
    for i in range(N):
        mn = 1e9
        for j in range(N):
            if i != j:
                mn = min(mn, np.hypot(x[i]-x[j], y[i]-y[j]))
        r[i] = 0.49 * mn
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    return x, y, r

def _rect_init(cols, rows):
    cells = []
    for j in range(rows):
        for i in range(cols):
            if len(cells) < N:
                x = (i + 0.5) / cols
                y = (j + 0.5) / rows
                cells.append([x, y])
    x = np.array([c[0] for c in cells])
    y = np.array([c[1] for c in cells])
    r = np.full(N, 0.5 / max(cols, rows) * 0.98)
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    return x, y, r

def _random_init(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.15, 0.85, N)
    y = rng.uniform(0.15, 0.85, N)
    r = np.full(N, 0.06)
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    return x, y, r

def _optimize(x, y, r, start):
    best = (x.copy(), y.copy(), r.copy())
    best_score = _sum_r(r)

    # L-BFGS-B with analytic gradient (fast, deep local search)
    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            p0 = np.concatenate([x, y, r])
            bounds = [(0, 1)] * (2*N) + [(0, 0.5)] * N
            res = minimize(
                _objective_grad, p0, args=(start,),
                jac=True, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass

    # Nelder-Mead for robustness
    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                _objective, p0, args=(start,),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass

    # Physics-based relaxation to escape local minima
    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            bx, by, br = _physics_optimize(best[0], best[1], best[2], start)
            bx, by, br = _make_valid(bx, by, br)
            if _is_valid(bx, by, br):
                s = _sum_r(br)
                if s > best_score:
                    best_score = s
                    best = (bx, by, br)
        except Exception:
            pass

    # Final high-precision L-BFGS-B polish with tight penalty
    if _elapsed(start) < TIME_LIMIT - 2:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            bounds = [(0, 1)] * (2*N) + [(0, 0.5)] * N
            res = minimize(
                _objective_grad, p0, args=(start,),
                jac=True, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 600, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass

    return best

def construct():
    start = time.time()
    inits = []
    inits.append(_hex_init())
    inits.append(_rect_init(6, 6))
    inits.append(_rect_init(7, 5))
    inits.append(_rect_init(5, 7))
    inits.append(_rect_init(8, 4))
    inits.append(_rect_init(4, 8))
    inits.append(_rect_init(9, 4))
    inits.append(_rect_init(4, 9))
    inits.append(_random_init(1))
    inits.append(_random_init(7))
    inits.append(_random_init(13))
    inits.append(_random_init(29))

    best = _make_valid(*inits[0])
    best_score = _sum_r(best[2])

    for k, (x, y, r) in enumerate(inits):
        if _elapsed(start) > TIME_LIMIT - 12:
            break
        res = _optimize(x, y, r, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best_score = s
                best = res

    # Final global polish on best
    if _elapsed(start) < TIME_LIMIT - 5:
        res = _optimize(*best, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best = res

    x, y, r = _make_valid(*best)
    return np.column_stack((x, y, r))
