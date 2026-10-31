"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 disjoint circles."""
import time
import numpy as np

N = 21
DEADLINE = 145.0  # safety margin before 150s kill


def _aspect_of(w, h):
    return w / max(h, 1e-9)


def _feasible(x, w, h, eps=1e-6):
    """x: (3N,) = [x0,y0,r0, ...]. Return (min_margin, min_radius, in_bounds)."""
    P = x.reshape(N, 3)
    xs, ys, rs = P[:, 0], P[:, 1], P[:, 2]
    # bounds: circle inside [0,w]x[0,h]
    m = np.minimum(np.minimum(xs - rs, w - xs - rs), np.minimum(ys - rs, h - ys - rs))
    # pairwise separation
    if N > 1:
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        s = rs[:, None] + rs[None, :]
        iu = np.triu_indices(N, 1)
        pm = d[iu] - s[iu]
        m = np.concatenate([m, pm])
    return float(m.min()), float(rs.min()), bool(np.all(rs > 0))


def _optimize_aspect(w, h, time_budget, rng):
    """For fixed w,h (perimeter 4), maximize sum r via L-BFGS-B. Returns best sum_r and x."""
    best_sum = -1.0
    best_x = None
    t0 = time.time()

    def objective(x):
        P = x.reshape(N, 3)
        rs = P[:, 2]
        # sum r
        val = rs.sum()
        # penalty for violations (soft) to guide gradient
        m, mr, ok = _feasible(x, w, h)
        pen = 0.0
        if m < 0:
            pen += 500.0 * m * m
        if mr < 1e-5:
            pen += 500.0 * (1e-5 - mr) ** 2
        return -(val - pen)

    # grid-ish initial guess scaled to fit
    def grid_init():
        cols = int(np.ceil(np.sqrt(N * w / h)))
        cols = max(1, min(cols, N))
        rows = int(np.ceil(N / cols))
        cw, ch = w / cols, h / rows
        pts = []
        for i in range(cols):
            for j in range(rows):
                if len(pts) >= N:
                    break
                pts.append(((i + 0.5) * cw, (j + 0.5) * ch, min(cw, ch) * 0.45))
        # pad if short
        while len(pts) < N:
            pts.append((w * 0.5, h * 0.5, 0.05))
        return np.array(pts).reshape(3 * N)

    # several restarts
    n_restarts = 6
    for k in range(n_restarts):
        if time.time() - t0 > time_budget:
            break
        if k == 0:
            x0 = grid_init()
        else:
            # perturb best or random
            if best_x is not None and rng.random() < 0.5:
                x0 = best_x + rng.normal(0, 0.02, 3 * N)
            else:
                x0 = grid_init() + rng.normal(0, 0.05, 3 * N)
        x0 = np.clip(x0, 1e-4, None)
        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 200, 'maxfun': 4000, 'ftol': 1e-9})
            xr = res.x
            m, mr, ok = _feasible(xr, w, h)
            if m < -1e-4:
                continue
            s = float(xr.reshape(N, 3)[:, 2].sum())
            if s > best_sum:
                best_sum = s
                best_x = xr.copy()
        except Exception:
            pass
    return best_sum, best_x


def construct():
    t_start = time.time()
    rng = np.random.default_rng(12345)
    best_val = -1.0
    best_out = None

    # sweep aspect ratios w/h; perimeter 2(w+h)=4 => w+h=2, h=2-w
    aspects = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0]
    # refine around good ones later
    per_aspect = 12.0

    for a in aspects:
        if time.time() - t_start > DEADLINE - 5:
            break
        w = 2.0 * a / (1.0 + a)
        h = 2.0 / (1.0 + a)
        budget = min(per_aspect, DEADLINE - 5 - (time.time() - t_start))
        if budget <= 0.5:
            break
        s, x = _optimize_aspect(w, h, budget, rng)
        if s > best_val:
            best_val = s
            best_out = (x, w, h)

    # refine best aspect with more restarts if time
    if best_out is not None and time.time() - t_start < DEADLINE - 3:
        x, w, h = best_out
        budget = DEADLINE - 3 - (time.time() - t_start)
        if budget > 2:
            s2, x2 = _optimize_aspect(w, h, budget, rng)
            if s2 > best_val:
                best_val = s2
                best_out = (x2, w, h)

    if best_out is None:
        # fallback: grid in a square
        x, w, h = grid_fallback()
        best_val = x[:, 2].sum()
        best_out = (x, w, h)

    x, w, h = best_out
    P = x.reshape(N, 3)
    # clamp tiny negatives / ensure positive radii
    P[:, 2] = np.clip(P[:, 2], 1e-6, None)
    # final feasibility check & repair: shrink if slightly infeasible
    for _ in range(50):
        m, mr, ok = _feasible(P, w, h)
        if m >= -1e-7:
            break
        # shrink all radii a bit
        P[:, 2] *= 0.99
    out = np.array([P[i, 0], P[i, 1], P[i, 2]] for i in range(N))
    return out


def grid_fallback():
    w = h = 1.0
    cols = 5
    rows = 5
    cw, ch = w / cols, h / rows
    pts = []
    for i in range(cols):
        for j in range(rows):
            if len(pts) >= N:
                break
            pts.append(((i + 0.5) * cw, (j + 0.5) * ch, min(cw, ch) * 0.45))
    while len(pts) < N:
        pts.append((0.5, 0.5, 0.05))
    return np.array(pts)


if __name__ == "__main__":
    res = construct()
    print("sum r =", res[:, 2].sum())
