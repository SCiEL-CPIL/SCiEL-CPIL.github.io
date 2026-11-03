"""Circle packing in unit square: physics-inspired optimization with annealing.

Structural change from the seed: instead of a single penalty-based L-BFGS/SLSQP
pass, we use a simulated-annealing-style outer loop over penalty weight λ,
combined with a custom gradient for the smooth penalty objective.  We also
use a better initial layout (hexagonal lattice) and a "grow radii" local
improvement step that exploits the fact that at a local optimum the radii are
constrained by the tightest contacts.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
DEADLINE = 138.0
START = time.time()


def _elapsed():
    return time.time() - START


def _smooth_max(a, b, k=20.0):
    """Smooth approximation of max(a,b)."""
    return np.where(np.abs(a - b) < 1e-8,
                    (a + b) / 2.0,
                    np.logaddexp(k * a, k * b) / k)


def _objective_and_grad(params, lam, k=20.0):
    """
    Objective: -sum(r) + lam * sum(violations^2)
    Returns (obj, grad) where grad is the gradient w.r.t. params.
    params: [x0,y0,r0, x1,y1,r1, ...]
    """
    n = len(params) // 3
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()

    # Clip for computation (bounds handled by L-BFGS-B)
    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    r = np.clip(r, 0.0, 0.5)

    # Boundary distances
    d_left = x
    d_right = 1.0 - x
    d_bot = y
    d_top = 1.0 - y
    d_bound = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bot, d_top))

    # Boundary violations: max(0, r - d_bound)
    bviol_raw = r - d_bound
    bviol = np.maximum(0.0, bviol_raw)

    # Pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)

    # Overlap violations: max(0, r_i + r_j - dist_ij)
    ov_raw = r[:, None] + r[None, :] - dist
    ov = np.maximum(0.0, ov_raw)

    total_viol_sq = np.sum(bviol ** 2) + np.sum(ov ** 2)
    obj = -np.sum(r) + lam * total_viol_sq

    # Gradient
    grad = np.zeros_like(params)

    # d/dx of -sum(r) is 0; d/dx of lam*sum(bviol^2):
    # bviol_i = max(0, r_i - d_bound_i)
    # d_bound_i = min(x_i, 1-x_i, y_i, 1-y_i)
    # If bviol_i > 0: dbviol/dx_i = -sign depending on which boundary is min
    for i in range(n):
        if bviol[i] > 0:
            # determine which boundary is active
            if d_bound[i] == d_left[i]:
                dbdx = 1.0
            elif d_bound[i] == d_right[i]:
                dbdx = -1.0
            else:
                dbdx = 0.0
            dbdy = 0.0
            if d_bound[i] == d_bot[i]:
                dbdy = 1.0
            elif d_bound[i] == d_top[i]:
                dbdy = -1.0
            # If two boundaries tie, average
            active = [d_left[i], d_right[i], d_bot[i], d_top[i]]
            signs_x = [1.0, -1.0, 0.0, 0.0]
            signs_y = [0.0, 0.0, 1.0, -1.0]
            min_val = d_bound[i]
            cnt = sum(1 for v in active if abs(v - min_val) < 1e-10)
            if cnt > 1:
                idxs = [j for j in range(4) if abs(active[j] - min_val) < 1e-10]
                dbdx = np.mean([signs_x[j] for j in idxs])
                dbdy = np.mean([signs_y[j] for j in idxs])
            # grad wrt x_i: lam * 2 * bviol_i * (-dbdx)
            grad[3 * i] += lam * 2.0 * bviol[i] * (-dbdx)
            grad[3 * i + 1] += lam * 2.0 * bviol[i] * (-dbdy)
            # grad wrt r_i: -1 + lam * 2 * bviol_i
            grad[3 * i + 2] += -1.0 + lam * 2.0 * bviol[i]

    # Pairwise overlap gradient
    for i in range(n):
        for j in range(i + 1, n):
            if ov[i, j] > 0:
                d = dist[i, j]
                if d < 1e-12:
                    continue
                # d(ov)/dx_i = (x_i - x_j)/d
                gx = (x[i] - x[j]) / d
                gy = (y[i] - y[j]) / d
                # grad wrt x_i: lam * 2 * ov * gx
                grad[3 * i] += lam * 2.0 * ov[i, j] * gx
                grad[3 * i + 1] += lam * 2.0 * ov[i, j] * gy
                # grad wrt x_j: lam * 2 * ov * (-gx)
                grad[3 * j] += lam * 2.0 * ov[i, j] * (-gx)
                grad[3 * j + 1] += lam * 2.0 * ov[i, j] * (-gy)
                # grad wrt r_i: -1 + lam * 2 * ov (from -sum(r))
                grad[3 * i + 2] += lam * 2.0 * ov[i, j]
                grad[3 * j + 2] += lam * 2.0 * ov[i, j]

    # Add the -1 from -sum(r) for each r
    grad[2::3] += -1.0

    return obj, grad


def _is_valid(x, y, r, eps=1e-7):
    d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    if np.any(r + eps > d):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    need = r[:, None] + r[None, :]
    if np.any(need + eps > dist):
        return False
    return True


def _fix_to_valid(x, y, r, max_iter=100):
    for _ in range(max_iter):
        if _is_valid(x, y, r):
            return x, y, r
        d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        boundary_viol = r - d
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        overlap_viol = r[:, None] + r[None, :] - dist

        circle_viol = np.zeros(N)
        for i in range(N):
            bv = max(0.0, boundary_viol[i])
            ov = np.max(np.maximum(0.0, overlap_viol[i, :]))
            circle_viol[i] = max(bv, ov)

        if np.max(circle_viol) <= 1e-12:
            break
        shrink = np.maximum(0.0, circle_viol) * 0.5
        r = r - shrink
        r = np.maximum(r, 1e-8)
    return x, y, r


def _hex_init(rng, spacing=None):
    """Hexagonal lattice init, returns x, y, r."""
    if spacing is None:
        spacing = 0.18
    x = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        row = i // 6
        col = i % 6
        x[i] = 0.12 + col * spacing + (0.5 * spacing if row % 2 else 0)
        y[i] = 0.12 + row * spacing * 0.866
    x = np.clip(x, 0.05, 0.95)
    y = np.clip(y, 0.05, 0.95)
    r = np.full(N, 0.42 * spacing)
    return x, y, r


def _grid_init(rng, cols=5, rows=5):
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    extra = [(0.5, 0.5)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    cell = min(1.0 / cols, 1.0 / rows)
    r = np.full(N, 0.4 * cell)
    return x, y, r


def _random_init(rng):
    x = rng.uniform(0.1, 0.9, size=N)
    y = rng.uniform(0.1, 0.9, size=N)
    r = rng.uniform(0.06, 0.14, size=N)
    return x, y, r


def _optimize_anneal(x, y, r, lam_schedule=None, max_iter_per_stage=200):
    """
    Annealing over penalty weight: start with small lam (focus on maximizing
    radii), then increase lam to enforce constraints.
    """
    if lam_schedule is None:
        lam_schedule = [1.0, 5.0, 20.0, 50.0, 100.0, 200.0]

    params = np.zeros(3 * N)
    params[0::3] = x
    params[1::3] = y
    params[2::3] = r

    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-4, 0.5)] * N

    for lam in lam_schedule:
        if _elapsed() > DEADLINE - 5.0:
            break

        def fun(p):
            obj, _ = _objective_and_grad(p, lam)
            return obj

        def jac(p):
            _, g = _objective_and_grad(p, lam)
            return g

        try:
            res = minimize(fun, params, jac=jac, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': max_iter_per_stage, 'ftol': 1e-12,
                                    'gtol': 1e-10, 'maxfun': max_iter_per_stage * 3})
            params = res.x
        except Exception:
            pass

    xx = np.clip(params[0::3], 0.0, 1.0)
    yy = np.clip(params[1::3], 0.0, 1.0)
    rr = np.clip(params[2::3], 1e-4, 0.5)
    return xx, yy, rr


def _grow_radii(x, y, r, steps=200, step_size=0.001):
    """
    Local improvement: try to grow each radius slightly, fixing positions.
    If a radius can't grow, try moving the circle.
    """
    for _ in range(steps):
        if _elapsed() > DEADLINE - 5.0:
            break
        improved = False
        for i in range(N):
            # Try to grow radius i
            trial_r = r.copy()
            trial_r[i] += step_size
            if _is_valid(x, y, trial_r):
                r = trial_r
                improved = True
            else:
                # Try nudging position
                for dx in [-step_size, step_size]:
                    for dy in [-step_size, step_size]:
                        nx = np.clip(x[i] + dx, 0.0, 1.0)
                        ny = np.clip(y[i] + dy, 0.0, 1.0)
                        x2 = x.copy()
                        y2 = y.copy()
                        x2[i] = nx
                        y2[i] = ny
                        if _is_valid(x2, y2, trial_r):
                            x, y = x2, y2
                            r = trial_r
                            improved = True
                            break
                    if improved:
                        break
        if not improved:
            step_size *= 0.95
            if step_size < 1e-6:
                break
    return x, y, r


def construct():
    best = None
    best_val = -np.inf
    rng = np.random.default_rng(12345)

    inits = [
        lambda: _hex_init(rng, spacing=0.18),
        lambda: _hex_init(rng, spacing=0.20),
        lambda: _hex_init(rng, spacing=0.16),
        lambda: _grid_init(rng, 5, 5),
        lambda: _grid_init(rng, 4, 7),
        lambda: _grid_init(rng, 6, 5),
        lambda: _random_init(rng),
    ]

    for init_fn in inits:
        if _elapsed() > DEADLINE - 15.0:
            break

        for restart in range(3):
            if _elapsed() > DEADLINE - 15.0:
                break

            x, y, r = init_fn()
            # Jitter
            x = np.clip(x + rng.normal(0, 0.008, N), 0.02, 0.98)
            y = np.clip(y + rng.normal(0, 0.008, N), 0.02, 0.98)
            r = np.clip(r + rng.normal(0, 0.004, N), 0.02, 0.35)

            # Annealed optimization
            xx, yy, rr = _optimize_anneal(x, y, r)
            xx, yy, rr = _fix_to_valid(xx, yy, rr)

            if _is_valid(xx, yy, rr):
                val = float(np.sum(rr))
                if val > best_val:
                    best_val = val
                    best = np.column_stack([xx, yy, rr])

            # Grow radii local improvement
            if _elapsed() < DEADLINE - 10.0:
                xx2, yy2, rr2 = _grow_radii(xx, yy, rr, steps=100, step_size=0.002)
                if _is_valid(xx2, yy2, rr2):
                    val2 = float(np.sum(rr2))
                    if val2 > best_val:
                        best_val = val2
                        best = np.column_stack([xx2, yy2, rr2])

            # One more anneal pass from the improved state
            if _elapsed() < DEADLINE - 15.0:
                xx3, yy3, rr3 = _optimize_anneal(xx2, yy2, rr2,
                                                 lam_schedule=[50.0, 100.0, 200.0, 500.0],
                                                 max_iter_per_stage=150)
                xx3, yy3, rr3 = _fix_to_valid(xx3, yy3, rr3)
                if _is_valid(xx3, yy3, rr3):
                    val3 = float(np.sum(rr3))
                    if val3 > best_val:
                        best_val = val3
                        best = np.column_stack([xx3, yy3, rr3])

                # Final grow
                if _elapsed() < DEADLINE - 8.0:
                    xx4, yy4, rr4 = _grow_radii(xx3, yy3, rr3, steps=80, step_size=0.001)
                    if _is_valid(xx4, yy4, rr4):
                        val4 = float(np.sum(rr4))
                        if val4 > best_val:
                            best_val = val4
                            best = np.column_stack([xx4, yy4, rr4])

    if best is not None:
        return best

    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    cells = cells[:N]
    while len(cells) < N:
        cells.append((0.5 / side, 0.5 / side, 0.1 / side))
    return np.array(cells[:N])
