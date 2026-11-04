"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
_DEADLINE = 110.0  # safety margin before 120s kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _soft_max_penalty(x, y, r, pairs, beta):
    """Soft penalty for overlap: sum over pairs of softmax of (r_i+r_j - dist)."""
    xi = x[pairs[:, 0]]
    yi = y[pairs[:, 0]]
    xj = x[pairs[:, 1]]
    yj = y[pairs[:, 1]]
    ri = r[pairs[:, 0]]
    rj = r[pairs[:, 1]]
    dx = xi - xj
    dy = yi - yj
    dist = np.sqrt(dx * dx + dy * dy + 1e-16)
    viol = ri + rj - dist
    # Soft-max: smooth upper bound on max violation
    m = np.max(viol)
    if m < -1e-12:
        return 0.0
    s = viol - m
    e = np.exp(s * beta)
    return (m + np.log(np.sum(e)) / beta)


def _soft_max_penalty_grad(x, y, r, pairs, beta):
    """Gradient of the soft-max penalty w.r.t. x, y, r."""
    xi = x[pairs[:, 0]]
    yi = y[pairs[:, 0]]
    xj = x[pairs[:, 1]]
    yj = y[pairs[:, 1]]
    ri = r[pairs[:, 0]]
    rj = r[pairs[:, 1]]
    dx = xi - xj
    dy = yi - yj
    dist = np.sqrt(dx * dx + dy * dy + 1e-16)
    viol = ri + rj - dist
    m = np.max(viol)
    if m < -1e-12:
        return np.zeros_like(x), np.zeros_like(y), np.zeros_like(r)
    s = viol - m
    e = np.exp(s * beta)
    w = e / np.sum(e)  # softmax weights, sum to 1
    # d(viol)/d(dist) = -1, d(dist)/d(xi) = dx/dist, etc.
    inv_dist = 1.0 / dist
    # Gradient contributions
    gx = np.zeros(N)
    gy = np.zeros(N)
    gr = np.zeros(N)
    # For each pair p with weight w_p:
    # d(viol_p)/d(x_i) = -dx_p * inv_dist_p  (since viol = ri+rj-dist, d(-dist)/dx_i = -dx/dist)
    # Wait: viol = ri + rj - dist, so d(viol)/d(x_i) = -d(dist)/d(x_i) = -(dx_i - dx_j)/dist = -dx/dist
    # d(viol)/d(x_j) = +dx/dist
    # d(viol)/d(r_i) = 1, d(viol)/d(r_j) = 1
    gx[pairs[:, 0]] -= w * dx * inv_dist
    gx[pairs[:, 1]] += w * dx * inv_dist
    gy[pairs[:, 0]] -= w * dy * inv_dist
    gy[pairs[:, 1]] += w * dy * inv_dist
    gr[pairs[:, 0]] += w
    gr[pairs[:, 1]] += w
    return gx, gy, gr


def _objective_and_grad(v, pairs, beta):
    """v = [x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]. Minimize -sum(r) + penalty."""
    n = len(pairs) * 2  # placeholder, use global N
    x = v[:N]
    y = v[N:2 * N]
    r = v[2 * N:3 * N]
    # Objective: -sum(r) + lambda * soft_max_penalty
    pen = _soft_max_penalty(x, y, r, pairs, beta)
    f = -np.sum(r) + 1000.0 * pen
    gx, gy, gr = _soft_max_penalty_grad(x, y, r, pairs, beta)
    grad = np.concatenate([
        -np.ones(N) + 1000.0 * gx,
        -np.ones(N) + 1000.0 * gy,
        -np.ones(N) + 1000.0 * gr,
    ])
    return f, grad


def _make_pairs():
    """All unique pairs."""
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j))
    return np.array(pairs, dtype=int)


def _validate_and_fix(arr):
    """Ensure circles are inside [0,1]^2 and non-overlapping (best effort)."""
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    # Clip radii to be positive and small
    r = np.clip(r, 1e-6, 0.5)
    # Clip positions so circles fit in square
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    return np.column_stack([x, y, r])


def _score(arr):
    """Compute sum of radii, returning 0 if invalid."""
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    # Check inside square
    if np.any(x < -1e-9) or np.any(x > 1 + 1e-9) or np.any(y < -1e-9) or np.any(y > 1 + 1e-9):
        return 0.0
    if np.any(r < 1e-9):
        return 0.0
    # Check non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0
    return float(np.sum(r))


def _grid_initial():
    """Grid of equal circles as starting point."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            rad = 0.5 / side - 1e-3
            cells.append([cx, cy, rad])
    return np.array(cells[:N])


def _random_initial(rng):
    """Random non-overlapping initial placement."""
    x = np.empty(N)
    y = np.empty(N)
    r = np.empty(N)
    for i in range(N):
        placed = False
        for attempt in range(2000):
            rad = rng.uniform(0.02, 0.15)
            cx = rng.uniform(rad, 1.0 - rad)
            cy = rng.uniform(rad, 1.0 - rad)
            ok = True
            for j in range(i):
                dx = cx - x[j]
                dy = cy - y[j]
                if np.sqrt(dx * dx + dy * dy) < rad + r[j] + 1e-4:
                    ok = False
                    break
            if ok:
                x[i] = cx
                y[i] = cy
                r[i] = rad
                placed = True
                break
        if not placed:
            # Fallback: small circle in a corner
            rad = 0.01
            x[i] = rad + 1e-3
            y[i] = rad + 1e-3
            r[i] = rad
    return np.column_stack([x, y, r])


def _optimize_from(arr, pairs, max_time=30.0):
    """Run L-BFGS-B optimization from initial arrangement."""
    x0 = np.concatenate([arr[:, 0], arr[:, 1], arr[:, 2]])
    bounds = [(1e-6, 1.0 - 1e-6)] * N + [(1e-6, 1.0 - 1e-6)] * N + [(1e-6, 0.5)] * N

    t0 = time.time()
    best_v = x0.copy()
    best_f = _objective_and_grad(x0, pairs, 50.0)[0]

    # Adaptive beta: start low, increase
    for beta in [20.0, 50.0, 100.0, 200.0]:
        if time.time() - t0 > max_time:
            break
        res = minimize(
            _objective_and_grad,
            x0,
            args=(pairs, beta),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8},
        )
        if res.success or res.fun < best_f:
            if res.fun < best_f:
                best_f = res.fun
                best_v = res.x.copy()
            x0 = res.x.copy()
        if time.time() - t0 > max_time:
            break

    # Convert back
    arr_opt = np.column_stack([best_v[:N], best_v[N:2 * N], best_v[2 * N:3 * N]])
    return arr_opt


def _polish(arr, pairs, max_time=15.0):
    """High-precision polish pass."""
    x0 = np.concatenate([arr[:, 0], arr[:, 1], arr[:, 2]])
    bounds = [(1e-7, 1.0 - 1e-7)] * N + [(1e-7, 1.0 - 1e-7)] * N + [(1e-7, 0.5)] * N
    t0 = time.time()
    for beta in [300.0, 500.0]:
        if time.time() - t0 > max_time:
            break
        res = minimize(
            _objective_and_grad,
            x0,
            args=(pairs, beta),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10},
        )
        if res.fun < _objective_and_grad(x0, pairs, beta)[0]:
            x0 = res.x.copy()
    arr_opt = np.column_stack([x0[:N], x0[N:2 * N], x0[2 * N:3 * N]])
    return arr_opt


def construct():
    global _START
    _START = time.time()
    _DEADLINE = _START + 110.0

    pairs = _make_pairs()

    # Start with grid
    best_arr = _grid_initial()
    best_score = _score(best_arr)

    # Try optimization from grid
    if _elapsed() < _DEADLINE - 60:
        arr_opt = _optimize_from(best_arr, pairs, max_time=40.0)
        arr_opt = _validate_and_fix(arr_opt)
        s = _score(arr_opt)
        if s > best_score:
            best_arr = arr_opt
            best_score = s

    # Random restarts
    rng = np.random.default_rng(42)
    n_restarts = 0
    while _elapsed() < _DEADLINE - 25:
        n_restarts += 1
        arr_init = _random_initial(rng)
        arr_opt = _optimize_from(arr_init, pairs, max_time=20.0)
        arr_opt = _validate_and_fix(arr_opt)
        s = _score(arr_opt)
        if s > best_score:
            best_arr = arr_opt
            best_score = s

    # Polish the best
    if _elapsed() < _DEADLINE - 5:
        polished = _polish(best_arr, pairs, max_time=15.0)
        polished = _validate_and_fix(polished)
        s = _score(polished)
        if s > best_score:
            best_arr = polished
            best_score = s

    # Final validation
    best_arr = _validate_and_fix(best_arr)
    final_score = _score(best_arr)
    if final_score <= 0:
        # Fallback to grid
        best_arr = _grid_initial()

    return best_arr
