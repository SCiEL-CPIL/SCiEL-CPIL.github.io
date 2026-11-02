"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE = 145.0  # safety margin before 150s kill
_IU = np.triu_indices(N, k=1)  # precomputed upper triangle indices


def _softplus(v, t=0.01):
    """Smooth approximation of max(0, v): t*log(1+exp(v/t))."""
    return t * np.log1p(np.exp(np.clip(v / t, -50, 50)))


def _compute_pairwise(x, y, r):
    """Compute pairwise distances and overlap amounts (vectorized)."""
    dx = x[_IU[0]] - x[_IU[1]]
    dy = y[_IU[0]] - y[_IU[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    ov = (r[_IU[0]] + r[_IU[1]]) - dist
    return dist, ov, dx, dy


def _penalty_objective(params, penalty=1000.0, t=0.01):
    """Penalty objective with smooth softplus for nonsmooth terms."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)

    score = -np.sum(r)

    # Wall boundary violations (smooth)
    wx = _softplus(r - x, t) + _softplus(r - (1.0 - x), t)
    wy = _softplus(r - y, t) + _softplus(r - (1.0 - y), t)
    wall = np.sum(wx) + np.sum(wy)

    # Pairwise overlap violations (smooth)
    _, ov, _, _ = _compute_pairwise(x, y, r)
    pair = np.sum(_softplus(ov, t))

    return score + penalty * (pair + wall)


def _penalty_grad(params, penalty=1000.0, t=0.01):
    """Analytic gradient of smooth penalty objective."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)
    n = N
    g = np.zeros_like(params)

    # d(-sum r)/dr = -1
    g[2::3] = -1.0

    # Wall contributions (smooth sigmoid weights)
    sig = lambda v: 1.0 / (1.0 + np.exp(-v / t))
    sx = sig(r - x)
    s1x = sig(r - (1.0 - x))
    sy = sig(r - y)
    s1y = sig(r - (1.0 - y))

    g[0::3] += penalty * (-sx + s1x)
    g[1::3] += penalty * (-sy + s1y)
    g[2::3] += penalty * (sx + s1x + sy + s1y)

    # Pairwise (smooth)
    dist, ov, dx, dy = _compute_pairwise(x, y, r)
    dist_safe = np.maximum(dist, 1e-12)
    w = sig(ov)  # sigmoid weight
    ux = dx / dist_safe
    uy = dy / dist_safe

    # Accumulate gradients using vectorized ops
    i_idx = _IU[0]
    j_idx = _IU[1]

    # For circle i: -w * ux in x, -w * uy in y, +w in r
    # For circle j: +w * ux in x, +w * uy in y, +w in r
    g[3 * i_idx] += penalty * (-w * ux)
    g[3 * j_idx] += penalty * (w * ux)
    g[3 * i_idx + 1] += penalty * (-w * uy)
    g[3 * j_idx + 1] += penalty * (w * uy)
    g[3 * i_idx + 2] += penalty * (w)
    g[3 * j_idx + 2] += penalty * (w)

    return g


def _grid_initial(seed):
    """Grid of equal circles with small jitter."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
    cells = cells[:N]
    arr = np.array(cells, dtype=float).reshape(-1)
    arr += rng.uniform(-1e-5, 1e-5, size=arr.size)
    return arr


def _random_initial(seed):
    """Random positions with modest radii."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.15, 0.85, size=N)
    y = rng.uniform(0.15, 0.85, size=N)
    r = np.full(N, 0.08)
    params = np.empty(3 * N)
    params[0::3] = x
    params[1::3] = y
    params[2::3] = r
    return params


def _bounds():
    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3*i] = 0.0
        hi[3*i] = 1.0
        lo[3*i+1] = 0.0
        hi[3*i+1] = 1.0
        lo[3*i+2] = 1e-6
        hi[3*i+2] = 0.5
    return (lo, hi)


def _stochastic_refine(params, deadline, rng, n_iters=8000):
    """Simulated annealing style local search with adaptive step."""
    val = _penalty_objective(params, penalty=3000.0, t=0.005)
    best = params.copy()
    best_val = val
    T = 0.02
    step = 0.012
    accept_count = 0
    total_count = 0

    for it in range(n_iters):
        if time.time() > deadline:
            break
        i = rng.randint(N)
        cand = params.copy()
        # Larger moves for position, smaller for radius
        cand[3*i] = params[3*i] + rng.uniform(-step, step)
        cand[3*i+1] = params[3*i+1] + rng.uniform(-step, step)
        cand[3*i+2] = params[3*i+2] + rng.uniform(-step*0.3, step*0.3)
        cand[3*i] = np.clip(cand[3*i], 0.0, 1.0)
        cand[3*i+1] = np.clip(cand[3*i+1], 0.0, 1.0)
        cand[3*i+2] = np.clip(cand[3*i+2], 1e-6, 0.5)

        new_val = _penalty_objective(cand, penalty=3000.0, t=0.005)
        d = new_val - val
        total_count += 1
        if d < 0 or rng.rand() < np.exp(-d / max(T, 1e-8)):
            params = cand
            val = new_val
            accept_count += 1
            if val < best_val:
                best_val = val
                best = params.copy()
        T *= 0.9995
        step *= 0.9998

        # Adaptive: if acceptance rate too low, increase step; if too high, decrease
        if total_count % 500 == 0 and total_count > 0:
            rate = accept_count / total_count
            if rate < 0.1:
                step *= 1.3
                T *= 1.5
            elif rate > 0.7:
                step *= 0.8
            accept_count = 0
            total_count = 0

    return best, best_val


def _enforce_validity(params):
    """Enforce all constraints: walls and pairwise non-overlap by optimal radius splitting."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    # Clamp to walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    # Iteratively resolve overlaps by splitting overlap proportionally
    for _ in range(800):
        dx = p[_IU[0], 0] - p[_IU[1], 0]
        dy = p[_IU[0], 1] - p[_IU[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[_IU[0], 2] + p[_IU[1], 2]) - dist
        active = overlap > 1e-12
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = _IU[0][k]
            j = _IU[1][k]
            ov = overlap[k]
            # Split overlap: each circle shrinks by half the overlap
            # but limited by its radius minus minimum
            shrink_i = min(ov * 0.5, p[i, 2] - 1e-6)
            shrink_j = min(ov - shrink_i, p[j, 2] - 1e-6)
            p[i, 2] -= shrink_i
            p[j, 2] -= shrink_j
        # Re-clamp to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    return p


def _score_valid(p):
    """Compute sum of radii for a valid configuration."""
    return np.sum(p[:, 2])


def _is_valid(p, tol=1e-7):
    """Check if configuration is valid (no overlaps, within bounds)."""
    p = p.reshape(N, 3)
    x, y, r = p[:, 0], p[:, 1], p[:, 2]
    # Wall check
    if np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(r < -tol) or np.any(r > 0.5 + tol):
        return False
    if np.any(x < r - tol) or np.any(x > 1 - r + tol) or np.any(y < r - tol) or np.any(y > 1 - r + tol):
        return False
    # Pairwise
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx*dx + dy*dy)
    if np.any((r[iu[0]] + r[iu[1]]) - dist > tol):
        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    lo, hi = _bounds()
    bounds_list = list(zip(lo, hi))

    best_params = None
    best_score = -np.inf

    # Multi-start with different initializations
    initializers = [
        lambda s: _grid_initial(s),
        lambda s: _random_initial(s),
    ]

    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828]

    # Phase 1: Multi-start optimization with penalty annealing
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        for init_fn in initializers:
            if time.time() > deadline - 10:
                break
            params = init_fn(seed)
            rng = np.random.RandomState(seed + 1000)

            # Penalty annealing: start with low penalty, increase gradually
            for pen in [50.0, 200.0, 1000.0]:
                if time.time() > deadline - 8:
                    break

                def obj(p, pen=pen):
                    return _penalty_objective(p, penalty=pen, t=0.01)

                def grad(p, pen=pen):
                    return _penalty_grad(p, penalty=pen, t=0.01)

                res = minimize(
                    obj, params, jac=grad, method='L-BFGS-B',
                    bounds=bounds_list,
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                params = res.x

                # Stochastic refinement after each penalty level
                params, val = _stochastic_refine(params, deadline - 8, rng, n_iters=4000)

                # Enforce validity and score
                valid_p = _enforce_validity(params)
                s = _score_valid(valid_p)
                if s > best_score:
                    best_score = s
                    best_params = valid_p

    # Phase 2: Final polish on best solution
    if best_params is not None and time.time() < deadline - 5:
        rng = np.random.RandomState(42)
        # High-precision local refinement
        params = best_params.reshape(-1).copy()
        for pen in [2000.0, 5000.0]:
            if time.time() > deadline - 3:
                break
            def obj(p, pen=pen):
                return _penalty_objective(p, penalty=pen, t=0.003)

            def grad(p, pen=pen):
                return _penalty_grad(p, penalty=pen, t=0.003)

            res = minimize(
                obj, params, jac=grad, method='L-BFGS-B',
                bounds=bounds_list,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            params = res.x

        params, val = _stochastic_refine(params, deadline - 2, rng, n_iters=5000)
        valid_p = _enforce_validity(params)
        s = _score_valid(valid_p)
        if s > best_score:
            best_score = s
            best_params = valid_p

    if best_params is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_params
