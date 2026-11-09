import time
import numpy as np
from scipy.optimize import minimize

N = 21
_DEADLINE = time.time() + 140.0  # safety margin under 150s kill


def _validate(arr):
    if arr is None or arr.shape != (N, 3):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    w, h, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r < 0):
        return None
    if np.any(w < 0) or np.any(w > 1) or np.any(h < 0) or np.any(h > 1):
        return None
    # non-overlap: dist >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(w[i] - w[j], h[i] - h[j])
            if d < r[i] + r[j] - 1e-7:
                return None
    return arr


def _rect_dims(a):
    # perimeter 4 => 2(w+h)=4 => w+h=2 ; aspect a = w/h
    h = 2.0 / (a + 1.0)
    w = 2.0 - h
    return w, h


def _smooth_penalty_overlap(pos, r, w, h, beta=50.0):
    """Smooth penalty for overlap and boundary violations using soft-min."""
    n = len(r)
    # Pairwise distances
    dx = pos[:, 0:1] - pos[0:1, 0].T
    dy = pos[:, 1:2] - pos[0:1, 1].T
    dist = np.sqrt(dx**2 + dy**2)
    # Required distance: r_i + r_j
    req = r[:, None] + r[None, :]
    # Soft-min of (dist - req) for i<j
    # Use log-sum-exp trick for soft-min
    diff = dist - req
    # Only consider upper triangle
    iu = np.triu_indices(n, k=1)
    diff_upper = diff[iu]
    # Soft-min: -log-sum-exp(-diff/beta) * beta
    # For numerical stability
    max_diff = np.max(diff_upper)
    if max_diff > 0:
        # All pairs are separated, penalty is small
        log_sum = np.sum(np.exp(-diff_upper / beta))
        soft_min = -beta * np.log(log_sum) + max_diff
        pen_overlap = soft_min
    else:
        log_sum = np.sum(np.exp(-diff_upper / beta))
        soft_min = -beta * np.log(log_sum) + max_diff
        pen_overlap = soft_min

    # Boundary violations
    pen_left = np.maximum(0, r - pos[:, 0])
    pen_right = np.maximum(0, r - (w - pos[:, 0]))
    pen_bottom = np.maximum(0, r - pos[:, 1])
    pen_top = np.maximum(0, r - (h - pos[:, 1]))

    total_pen = pen_overlap + np.sum(pen_left) + np.sum(pen_right) + np.sum(pen_bottom) + np.sum(pen_top)
    return total_pen


def _solve_aspect_smooth(a, x0, rng, time_left_fn):
    """Solve using smooth penalty formulation."""
    w, h = _rect_dims(a)
    n = N

    def objective(x):
        pos = x[:2 * n].reshape(n, 2)
        r = x[2 * n:]
        # Penalize overlaps and boundary violations
        pen = _smooth_penalty_overlap(pos, r, w, h, beta=30.0)
        return -np.sum(r) + 10.0 * pen

    def jac(x):
        pos = x[:2 * n].reshape(n, 2)
        r = x[2 * n:]
        # Numerical gradient for simplicity
        eps = 1e-7
        grad = np.zeros_like(x)
        f0 = objective(x)
        for i in range(len(x)):
            x[i] += eps
            f1 = objective(x)
            x[i] -= eps
            grad[i] = (f1 - f0) / eps
        return grad

    bounds = [(0.0, w)] * n + [(0.0, h)] * n + [(1e-4, 1.0)] * n

    def time_left_check():
        return time_left_fn() < 0

    try:
        res = minimize(objective, x0, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    except Exception:
        return None

    if time_left_check():
        return None

    x = res.x
    arr = np.zeros((n, 3))
    arr[:, 0] = x[:2 * n:2]
    arr[:, 1] = x[1:2 * n:2]
    arr[:, 2] = x[2 * n:]

    # Clip to valid range
    arr[:, 0] = np.clip(arr[:, 0], 0, w)
    arr[:, 1] = np.clip(arr[:, 1], 0, h)
    arr[:, 2] = np.clip(arr[:, 2], 1e-4, 1.0)

    v = _validate(arr)
    if v is not None:
        return v
    return None


def _polish(arr, time_left_fn):
    """Final high-precision polish pass using SLSQP with tight tolerances."""
    n = N
    w = arr[:, 0].max() + arr[:, 2].max()
    h = arr[:, 1].max() + arr[:, 2].max()
    # Actually, we need to work in the original aspect ratio space
    # Extract aspect from current solution
    # The bounding box should have perimeter 4
    x_max = np.max(arr[:, 0] + arr[:, 2])
    y_max = np.max(arr[:, 1] + arr[:, 2])
    x_min = np.min(arr[:, 0] - arr[:, 2])
    y_min = np.min(arr[:, 1] - arr[:, 2])
    rect_w = x_max - x_min
    rect_h = y_max - y_min
    # Normalize to perimeter 4
    scale = 4.0 / (2 * (rect_w + rect_h)) if (rect_w + rect_h) > 0 else 1.0
    rect_w *= scale
    rect_h *= scale

    if time_left_fn() < 2:
        return arr

    # Work in normalized [0,1]x[0,1] coordinates for polish
    x0 = np.empty(3 * n)
    x0[:2 * n:2] = (arr[:, 0] - x_min) / rect_w
    x0[1:2 * n:2] = (arr[:, 1] - y_min) / rect_h
    x0[2 * n:] = arr[:, 2] / scale

    def objective(x):
        return -np.sum(x[2 * n:])

    def jac(x):
        g = np.zeros(3 * n)
        g[2 * n:] = -1.0
        return g

    def con_overlap(x):
        pos = x[:2 * n].reshape(n, 2)
        r = x[2 * n:]
        vals = np.empty(n * (n - 1) // 2)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                d2 = (pos[i, 0] - pos[j, 0]) ** 2 + (pos[i, 1] - pos[j, 1]) ** 2
                s = r[i] + r[j]
                vals[idx] = d2 - s * s
                idx += 1
        return vals

    def con_left(x):
        return x[:2 * n:2] - x[2 * n:]

    def con_right(x):
        return (1.0 - x[:2 * n:2]) - x[2 * n:]

    def con_bottom(x):
        return x[1:2 * n:2] - x[2 * n:]

    def con_top(x):
        return (1.0 - x[1:2 * n:2]) - x[2 * n:]

    cons = [
        {"type": "ineq", "fun": con_overlap},
        {"type": "ineq", "fun": con_left},
        {"type": "ineq", "fun": con_right},
        {"type": "ineq", "fun": con_bottom},
        {"type": "ineq", "fun": con_top},
    ]
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(1e-4, 1.0)] * n

    try:
        res = minimize(objective, x0, jac=jac, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-14})
    except Exception:
        return arr

    if time_left_fn() < 0:
        return arr

    if res.fun < -np.sum(arr[:, 2]) * 0.99:
        x = res.x
        new_arr = np.zeros((n, 3))
        new_arr[:, 0] = x[:2 * n:2] * rect_w + x_min
        new_arr[:, 1] = x[1:2 * n:2] * rect_h + y_min
        new_arr[:, 2] = x[2 * n:] * scale

        # Validate in original coordinate system
        v = _validate(new_arr)
        if v is not None and np.sum(v[:, 2]) > np.sum(arr[:, 2]):
            return v
    return arr


def _seed_grid(a, rng):
    # a simple grid-like feasible-ish start for the aspect
    w, h = _rect_dims(a)
    cols = int(np.ceil(np.sqrt(N * w / h)))
    rows = int(np.ceil(N / cols))
    xs = np.linspace(0.1, 0.9, cols) * w
    ys = np.linspace(0.1, 0.9, rows) * h
    x0 = np.empty(3 * N)
    for k in range(N):
        i = k % cols
        j = k // cols
        x0[2 * k] = xs[i]
        x0[2 * k + 1] = ys[j]
        x0[2 * N + k] = 0.05
    return x0


def construct():
    t_start = time.time()
    rng = np.random.default_rng(12345)
    best = None
    best_val = -np.inf

    def time_left_fn():
        return _DEADLINE - time.time()

    aspects = [1.0, 1.3, 0.77, 1.6, 0.62, 1.15, 0.87]
    starts_per_aspect = 2

    for a in aspects:
        for s in range(starts_per_aspect):
            if time_left_fn() < 5:
                break
            # mix: one near-grid start, rest random
            if s == 0:
                x0 = _seed_grid(a, rng)
            else:
                w, h = _rect_dims(a)
                x0 = np.empty(3 * N)
                for k in range(N):
                    x0[2 * k] = rng.uniform(0.05, 0.95) * w
                    x0[2 * k + 1] = rng.uniform(0.05, 0.95) * h
                    x0[2 * N + k] = rng.uniform(0.02, 0.12)

            arr = _solve_aspect_smooth(a, x0, rng, time_left_fn)
            if arr is not None:
                val = np.sum(arr[:, 2])
                if val > best_val:
                    best_val = val
                    best = arr
        if time_left_fn() < 5:
            break

    # Final polish on best solution
    if best is not None and time_left_fn() > 5:
        best = _polish(best, time_left_fn)

    if best is None:
        # fallback: seed-like uniform grid (always feasible)
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return best
