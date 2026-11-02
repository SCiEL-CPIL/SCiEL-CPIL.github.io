"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE = 145.0  # safety margin before 150s kill

_IU = np.triu_indices(N, k=1)  # precomputed upper triangle indices


def _penalty_objective(params, penalty=1000.0):
    """Penalty objective: minimize -sum(r) + penalty*(overlaps + wall violations)."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)

    score = -np.sum(r)

    # Wall boundary violations
    wx = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x))
    wy = np.maximum(0.0, r - y) + np.maximum(0.0, r - (1.0 - y))
    wall = np.sum(wx) + np.sum(wy)

    # Pairwise overlap violations
    dx = x[_IU[0]] - x[_IU[1]]
    dy = y[_IU[0]] - y[_IU[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap = np.maximum(0.0, (r[_IU[0]] + r[_IU[1]]) - dist)
    pair = np.sum(overlap)

    return score + penalty * (pair + wall)


def _penalty_grad(params, penalty=1000.0):
    """Analytic gradient of penalty objective (fully vectorized)."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = np.maximum(p[:, 2], 1e-6)
    n = N
    g = np.zeros_like(params)

    # d(-sum r)/dr = -1
    g[2::3] = -1.0

    # Wall contributions (vectorized)
    # Left wall: r - x > 0
    active_left = r - x > 0
    g[0::3][active_left] += penalty * (-1.0)
    g[2::3][active_left] += penalty * (+1.0)
    # Right wall: r - (1-x) > 0
    active_right = r - (1.0 - x) > 0
    g[0::3][active_right] += penalty * (+1.0)
    g[2::3][active_right] += penalty * (+1.0)
    # Bottom wall: r - y > 0
    active_bottom = r - y > 0
    g[1::3][active_bottom] += penalty * (-1.0)
    g[2::3][active_bottom] += penalty * (+1.0)
    # Top wall: r - (1-y) > 0
    active_top = r - (1.0 - y) > 0
    g[1::3][active_top] += penalty * (+1.0)
    g[2::3][active_top] += penalty * (+1.0)

    # Pairwise (vectorized)
    dx = x[_IU[0]] - x[_IU[1]]
    dy = y[_IU[0]] - y[_IU[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    dist_safe = np.maximum(dist, 1e-10)
    ov = (r[_IU[0]] + r[_IU[1]]) - dist
    active = ov > 0
    if np.any(active):
        ax = _IU[0][active]
        ay = _IU[1][active]
        adx = dx[active] / dist_safe[active]
        ady = dy[active] / dist_safe[active]
        # Use np.add.at for accumulation
        np.add.at(g[0::3], ax, penalty * (-adx))
        np.add.at(g[0::3], ay, penalty * (+adx))
        np.add.at(g[1::3], ax, penalty * (-ady))
        np.add.at(g[1::3], ay, penalty * (+ady))
        np.add.at(g[2::3], ax, penalty * 1.0)
        np.add.at(g[2::3], ay, penalty * 1.0)

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


def _hex_initial(seed):
    """Hexagonal packing initialization - denser than square grid."""
    rng = np.random.RandomState(seed)
    # Hexagonal packing: rows offset by half spacing
    # For 26 circles, try a layout like 5-6-5-6-4 or similar
    # Use a hexagonal lattice and select N points
    spacing = 0.21  # approximate spacing for 26 circles
    sqrt3 = np.sqrt(3)
    points = []
    for row in range(8):
        y_pos = row * spacing * sqrt3 / 2
        if y_pos > 0.95:
            break
        offset = (row % 2) * spacing / 2
        for col in range(8):
            x_pos = col * spacing + offset
            if x_pos > 0.95:
                break
            points.append((x_pos, y_pos))
    
    # If we have enough points, use them; otherwise pad with jittered positions
    if len(points) >= N:
        # Select N points that are most spread out
        rng.shuffle(points)
        points = points[:N]
    else:
        # Pad with random positions
        while len(points) < N:
            points.append((rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)))
    
    # Estimate radius from minimum distance
    pts = np.array(points)
    min_dist = np.inf
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            d = np.sqrt((pts[i,0]-pts[j,0])**2 + (pts[i,1]-pts[j,1])**2)
            min_dist = min(min_dist, d)
    base_r = min_dist / 2 - 0.005
    
    arr = np.zeros(3 * N)
    for i in range(N):
        arr[3*i] = points[i][0]
        arr[3*i+1] = points[i][1]
        arr[3*i+2] = base_r
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


def _stochastic_refine(params, deadline, rng, n_iters=6000):
    """Simulated annealing style local search (optimized)."""
    pen = 5000.0
    val = _penalty_objective(params, pen)
    best = params.copy()
    best_val = val
    T = 0.03
    step = 0.015
    exp_arg_max = 50.0  # cap for exp to avoid overflow

    for it in range(n_iters):
        if time.time() > deadline:
            break
        i = rng.randint(N)
        cand = params.copy()
        cand[3*i] = params[3*i] + rng.uniform(-step, step)
        cand[3*i+1] = params[3*i+1] + rng.uniform(-step, step)
        cand[3*i+2] = params[3*i+2] + rng.uniform(-step*0.4, step*0.4)
        cand[3*i] = np.clip(cand[3*i], 0.0, 1.0)
        cand[3*i+1] = np.clip(cand[3*i+1], 0.0, 1.0)
        cand[3*i+2] = np.clip(cand[3*i+2], 1e-6, 0.5)

        new_val = _penalty_objective(cand, pen)
        d = new_val - val
        if d < 0:
            params = cand
            val = new_val
            if val < best_val:
                best_val = val
                best = params.copy()
        else:
            arg = -d / max(T, 1e-8)
            if arg > exp_arg_max:
                accept = False
            else:
                accept = rng.rand() < np.exp(arg)
            if accept:
                params = cand
                val = new_val
                if val < best_val:
                    best_val = val
                    best = params.copy()
        T *= 0.9992
        step *= 0.9997

    return best, best_val


def _enforce_validity(params):
    """Enforce all constraints: walls and pairwise non-overlap by shrinking radii."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    # Clamp to walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    # Iteratively resolve overlaps by shrinking the smaller circle
    for _ in range(500):
        dx = p[_IU[0], 0] - p[_IU[1], 0]
        dy = p[_IU[0], 1] - p[_IU[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[_IU[0], 2] + p[_IU[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        ax = _IU[0][active]
        ay = _IU[1][active]
        ov = overlap[active]
        for k in range(len(ax)):
            i = ax[k]
            j = ay[k]
            o = ov[k]
            if p[i, 2] <= p[j, 2]:
                shrink = min(o, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = o - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(o, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = o - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
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


def _polish(params, deadline):
    """Final high-precision polish with tight tolerances."""
    lo, hi = _bounds()
    bounds_list = list(zip(lo, hi))
    
    # Use a high penalty for the polish
    pen = 10000.0
    
    def obj(p):
        return _penalty_objective(p, penalty=pen)
    
    def grad(p):
        return _penalty_grad(p, penalty=pen)
    
    try:
        res = minimize(
            obj, params, jac=grad, method='L-BFGS-B',
            bounds=bounds_list,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        return res.x
    except Exception:
        return params


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
        lambda s: _hex_initial(s),
    ]

    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828]

    for seed in seeds:
        if time.time() > deadline - 5:
            break
        for init_fn in initializers:
            if time.time() > deadline - 5:
                break
            params = init_fn(seed)
            rng = np.random.RandomState(seed + 1000)

            # Multi-penalty L-BFGS-B
            for pen in [100.0, 500.0, 2000.0]:
                if time.time() > deadline - 3:
                    break

                def obj(p, pen=pen):
                    return _penalty_objective(p, penalty=pen)

                def grad(p, pen=pen):
                    return _penalty_grad(p, penalty=pen)

                res = minimize(
                    obj, params, jac=grad, method='L-BFGS-B',
                    bounds=bounds_list,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                params = res.x

                # Stochastic refinement
                params, val = _stochastic_refine(params, deadline - 3, rng)

                # Enforce validity and score
                valid_p = _enforce_validity(params)
                s = _score_valid(valid_p)
                if s > best_score:
                    best_score = s
                    best_params = valid_p

    # Final polish pass on the best solution
    if best_params is not None and time.time() < deadline - 2:
        polished = _polish(best_params.reshape(-1), deadline)
        valid_p = _enforce_validity(polished)
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
