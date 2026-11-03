"""Circle packing in perimeter-4 rectangle: gradient-based optimization of
positions + aspect ratio, starting from a grid seed."""
import time
import numpy as np

N = 21
DEADLINE = 145.0  # safety margin under 150s wall-clock


def _grid_init():
    # 5x5 grid on a square (w=h=1), 21 circles, tiny epsilon so it's valid.
    side = 5
    eps = 1e-4
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= N:
                break
            pts.append((0.5 + i, 0.5 + j, 0.5 - eps))
    pts = pts[:N]
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    r = np.array([p[2] for p in pts])
    return x, y, r


def _evaluate(params, w):
    # params: x(N), y(N), r(N)
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    h = 2.0 - w
    # objective: sum of radii
    score = np.sum(r)
    # soft penalties for overlaps and walls (so L-BFGS can navigate)
    pen = 0.0
    # overlaps: for each pair, require dist >= r_i + r_j
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(x[i]-x[j], y[i]-y[j])
            need = r[i] + r[j]
            if d < need:
                pen += (need - d)**2
    # walls: r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i
    for i in range(N):
        for m in (x[i] - r[i], w - x[i] - r[i], y[i] - r[i], h - y[i] - r[i]):
            if m < 0:
                pen += m*m
    return score, pen


def _validity(x, y, r, w):
    h = 2.0 - w
    eps = 1e-7
    for i in range(N):
        if r[i] <= 0:
            return False
        if x[i] - r[i] < -eps or w - x[i] - r[i] < -eps:
            return False
        if y[i] - r[i] < -eps or h - y[i] - r[i] < -eps:
            return False
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(x[i]-x[j], y[i]-y[j])
            if d < r[i] + r[j] - eps:
                return False
    return True


def _best_known(x, y, r, w):
    # return (score, x, y, r, w) if valid
    if _validity(x, y, r, w):
        return np.sum(r), x.copy(), y.copy(), r.copy(), w
    return None


def construct():
    t0 = time.time()
    x0, y0, r0 = _grid_init()
    w0 = 1.0
    base_score = np.sum(r0)
    best = (base_score, x0.copy(), y0.copy(), r0.copy(), w0)

    from scipy.optimize import minimize

    # We optimize over (x, y, r, w). Use a barrier/penalty objective.
    # To keep r positive, work with log-r? Simpler: bound r in [0, 0.5].
    def objective(p):
        if time.time() > DEADLINE:
            raise RuntimeError("deadline")
        w = p[-1]
        x = p[:N]
        y = p[N:2*N]
        r = p[2*N:3*N]
        score, pen = _evaluate(p, w)
        # large weight on penalty so validity dominates, but still push score
        return pen*1000.0 - score

    def jac(p):
        # numerical gradient (simple, robust)
        if time.time() > DEADLINE:
            raise RuntimeError("deadline")
        g = np.zeros_like(p)
        f0 = objective(p)
        for i in range(len(p)):
            h = 1e-6 * (abs(p[i]) + 1)
            pp = p.copy()
            pp[i] += h
            g[i] = (objective(pp) - f0) / h
        return g

    p0 = np.concatenate([x0, y0, r0, [w0]])
    bounds = []
    for i in range(N):
        bounds.append((0.0, 2.0))      # x
    for i in range(N):
        bounds.append((0.0, 2.0))      # y
    for i in range(N):
        bounds.append((0.0, 1.0))      # r
    bounds.append((0.3, 1.7))          # w (keep aspect reasonable)

    try:
        res = minimize(objective, p0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 300, 'maxfun': 6000,
                                'ftol': 1e-10, 'gtol': 1e-7})
    except Exception:
        res = None

    if res is not None and res.x is not None:
        w = res.x[-1]
        x = res.x[:N]
        y = res.x[N:2*N]
        r = res.x[2*N:3*N]
        cand = _best_known(x, y, r, w)
        if cand is not None and cand[0] > best[0]:
            best = cand

    # If the optimized result is invalid or not better, keep the grid.
    # Return as (n,3) array: x, y, r — but we must ensure the bounding
    # rectangle has perimeter <= 4. Our w,h satisfy w+h=2 -> perimeter 4.
    # The evaluator will use the circles' actual bounding box; we return
    # x, y, r as-is (positions within [0,w]x[0,h]).
    if time.time() > DEADLINE:
        pass
    bx = best[1]; by = best[2]; br = best[3]
    return np.column_stack([bx, by, br])


if __name__ == "__main__":
    out = construct()
    print(out.shape, out[:3])
