import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-10
DEADLINE = 145.0  # safety margin before 150s kill


def _validate_and_score(params):
    """Validate configuration and return (is_valid, score)."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]

    # Check walls
    if np.any(x < -EPS) or np.any(x > 1 + EPS):
        return False, 0.0
    if np.any(y < -EPS) or np.any(y > 1 + EPS):
        return False, 0.0
    if np.any(r < -EPS):
        return False, 0.0

    # Check circle fits in square
    if np.any(r - x < -EPS) or np.any(r - (1 - x) < -EPS):
        return False, 0.0
    if np.any(r - y < -EPS) or np.any(r - (1 - y) < -EPS):
        return False, 0.0

    # Check pairwise distances
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any(dist < r[iu[0]] + r[iu[1]] - EPS):
        return False, 0.0

    return True, np.sum(r)


def _fix_config(params):
    """Fix a configuration to be valid by clipping and shrinking."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-8, 0.5)

    # Clamp radii to fit walls
    for i in range(N):
        max_r = min(p[i, 0], 1 - p[i, 0], p[i, 1], 1 - p[i, 1])
        p[i, 2] = min(p[i, 2], max(max_r, 1e-8))

    # Resolve overlaps iteratively
    for _ in range(50):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = p[iu[0], 2] + p[iu[1], 2] - dist
        active = overlap > 1e-12
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            # Split overlap: shrink both circles
            shrink_i = ov * 0.5
            shrink_j = ov * 0.5
            p[i, 2] = max(p[i, 2] - shrink_i, 1e-8)
            p[j, 2] = max(p[j, 2] - shrink_j, 1e-8)
        # Re-clamp to walls
        for i in range(N):
            max_r = min(p[i, 0], 1 - p[i, 0], p[i, 1], 1 - p[i, 1])
            p[i, 2] = min(p[i, 2], max(max_r, 1e-8))

    return p


def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    return np.where(x > 20.0 / beta, x, (1.0 / beta) * np.log(1.0 + np.exp(beta * x)))


def _objective(params, beta=50.0):
    """Penalty objective: minimize -sum(r) + penalty*(overlaps + wall violations)."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]

    score = -np.sum(r)

    # Wall violations
    wx = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x))
    wy = np.maximum(0.0, r - y) + np.maximum(0.0, r - (1.0 - y))
    wall = np.sum(wx) + np.sum(wy)

    # Pairwise overlap violations
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap = np.maximum(0.0, (r[iu[0]] + r[iu[1]]) - dist)
    pair = np.sum(overlap)

    return score + 1000.0 * (pair + wall)


def _objective_grad(params):
    """Gradient of the penalty objective."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    g = np.zeros_like(params)

    g[2::3] = -1.0

    # Wall contributions
    penalty = 1000.0
    for i in range(N):
        if r[i] - x[i] > 0:
            g[3*i] += penalty * (-1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - (1.0 - x[i]) > 0:
            g[3*i] += penalty * (+1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - y[i] > 0:
            g[3*i+1] += penalty * (-1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - (1.0 - y[i]) > 0:
            g[3*i+1] += penalty * (+1.0)
            g[3*i+2] += penalty * (+1.0)

    # Pairwise
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    dist_safe = np.maximum(dist, 1e-10)
    ov = (r[iu[0]] + r[iu[1]]) - dist
    active = ov > 0
    if np.any(active):
        ax = iu[0][active]
        ay = iu[1][active]
        adx = dx[active] / dist_safe[active]
        ady = dy[active] / dist_safe[active]
        for k in range(len(ax)):
            i = ax[k]
            j = ay[k]
            g[3*i] += penalty * (-adx[k])
            g[3*j] += penalty * (+adx[k])
            g[3*i+1] += penalty * (-ady[k])
            g[3*j+1] += penalty * (+ady[k])
            g[3*i+2] += penalty * (+1.0)
            g[3*j+2] += penalty * (+1.0)

    return g


def _get_delaunay_edges(p):
    """Get Delaunay triangulation edges for contact graph."""
    try:
        pts = p[:, :2]
        delaunay = Delaunay(pts)
        edges = set()
        for simplex in delaunay.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        # Fallback: all pairs
        return [(i, j) for i in range(N) for j in range(i+1, N)]


def _delaunay_objective(params, edges, beta=50.0):
    """Objective using only Delaunay edges for speed."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]

    score = -np.sum(r)

    # Wall violations
    wx = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x))
    wy = np.maximum(0.0, r - y) + np.maximum(0.0, r - (1.0 - y))
    wall = np.sum(wx) + np.sum(wy)

    # Pairwise overlap violations (only Delaunay edges)
    pair = 0.0
    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = max(0.0, r[i] + r[j] - dist)
        pair += overlap

    return score + 1000.0 * (pair + wall)


def _delaunay_objective_grad(params, edges):
    """Gradient of Delaunay objective."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    g = np.zeros_like(params)
    g[2::3] = -1.0
    penalty = 1000.0

    # Wall contributions
    for i in range(N):
        if r[i] - x[i] > 0:
            g[3*i] += penalty * (-1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - (1.0 - x[i]) > 0:
            g[3*i] += penalty * (+1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - y[i] > 0:
            g[3*i+1] += penalty * (-1.0)
            g[3*i+2] += penalty * (+1.0)
        if r[i] - (1.0 - y[i]) > 0:
            g[3*i+1] += penalty * (+1.0)
            g[3*i+2] += penalty * (+1.0)

    # Pairwise (Delaunay edges only)
    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < 1e-10:
            continue
        ov = r[i] + r[j] - dist
        if ov > 0:
            adx = dx / dist
            ady = dy / dist
            g[3*i] += penalty * (-adx)
            g[3*j] += penalty * (+adx)
            g[3*i+1] += penalty * (-ady)
            g[3*j+1] += penalty * (+ady)
            g[3*i+2] += penalty * (+1.0)
            g[3*j+2] += penalty * (+1.0)

    return g


def _initial_grid(seed):
    """Grid initialization."""
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


def _initial_hex(seed):
    """Hexagonal initialization."""
    rng = np.random.RandomState(seed)
    points = []
    for i in range(7):
        for j in range(7):
            x = i
            y = j * 1.5 + (0.5 if i % 2 else 0)
            points.append((x, y))
    points = np.array(points)
    points -= np.mean(points, axis=0)
    max_coord = np.max(np.abs(points))
    scale = 0.45 / max_coord
    points *= scale
    points += 0.5
    dists = np.sum((points - 0.5)**2, axis=1)
    indices = np.argsort(dists)[:N]
    p = points[indices]
    radii = np.full(N, 0.05)
    params = np.empty(3 * N)
    params[0::3] = p[:, 0]
    params[1::3] = p[:, 1]
    params[2::3] = radii
    params += rng.uniform(-1e-4, 1e-4, size=params.size)
    return params


def _initial_corner_weighted(seed):
    """Initialize with circles near corners and edges."""
    rng = np.random.RandomState(seed)
    corners = [(0.05, 0.05), (0.95, 0.05), (0.05, 0.95), (0.95, 0.95)]
    edges = [(0.5, 0.05), (0.5, 0.95), (0.05, 0.5), (0.95, 0.5)]
    center = [(0.5, 0.5)]
    
    points = []
    for c in corners:
        points.append((c[0] + rng.uniform(-0.02, 0.02), c[1] + rng.uniform(-0.02, 0.02)))
    for e in edges:
        points.append((e[0] + rng.uniform(-0.02, 0.02), e[1] + rng.uniform(-0.02, 0.02)))
    points.append((0.5 + rng.uniform(-0.1, 0.1), 0.5 + rng.uniform(-0.1, 0.1)))
    
    # Fill remaining with random points
    while len(points) < N:
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        points.append((x, y))
    
    points = np.array(points[:N])
    radii = np.full(N, 0.06)
    
    params = np.empty(3 * N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    params += rng.uniform(-1e-4, 1e-4, size=params.size)
    return params


def _initial_random(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = rng.uniform(0.03, 0.08, N)
    params = np.empty(3 * N)
    params[0::3] = x
    params[1::3] = y
    params[2::3] = r
    return params


def _force_optimize(params, time_limit=10.0, seed=42):
    """Force-directed optimization to resolve overlaps and maximize radii."""
    best_p = params.reshape(N, 3).copy()
    best_score = _validate_and_score(best_p.flatten())[1]
    rng = np.random.RandomState(seed)
    start = time.time()

    # Adaptive parameters
    lr = 0.01
    decay = 0.995
    max_iter = 5000

    for it in range(max_iter):
        if time.time() - start > time_limit:
            break

        p = best_p.copy()
        x = p[:, 0]
        y = p[:, 1]
        r = p[:, 2]

        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)

        # Wall repulsion
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += 10.0 * (r[i] - x[i])
                fr[i] -= 5.0 * (r[i] - x[i])
            if x[i] > 1 - r[i]:
                fx[i] -= 10.0 * (x[i] - (1 - r[i]))
                fr[i] -= 5.0 * (x[i] - (1 - r[i]))
            if y[i] < r[i]:
                fy[i] += 10.0 * (r[i] - y[i])
                fr[i] -= 5.0 * (r[i] - y[i])
            if y[i] > 1 - r[i]:
                fy[i] -= 10.0 * (y[i] - (1 - r[i]))
                fr[i] -= 5.0 * (y[i] - (1 - r[i]))

        # Pairwise repulsion
        iu = np.triu_indices(N, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = np.maximum(dist, 1e-8)
        overlap = r[iu[0]] + r[iu[1]] - dist

        for k in range(len(iu[0])):
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            if ov > 0:
                force = 5.0 * ov
                adx = dx[k] / dist_safe[k]
                ady = dy[k] / dist_safe[k]
                fx[i] -= force * adx
                fx[j] += force * adx
                fy[i] -= force * ady
                fy[j] += force * ady
                fr[i] -= force * 0.5
                fr[j] -= force * 0.5

        # Pressure to increase radii
        fr += 0.1

        # Update
        p[:, 0] += lr * fx
        p[:, 1] += lr * fy
        p[:, 2] += lr * fr

        # Clip
        p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
        p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
        p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

        # Clamp to walls
        for i in range(N):
            max_r = min(p[i, 0], 1 - p[i, 0], p[i, 1], 1 - p[i, 1])
            p[i, 2] = min(p[i, 2], max(max_r, 1e-6))

        # Fix and score
        p_fixed = _fix_config(p.flatten()).reshape(N, 3)
        score = _validate_and_score(p_fixed.flatten())[1]

        if score > best_score + 1e-10:
            best_score = score
            best_p = p_fixed

        lr *= decay

    return best_p


def _lbfgs_optimize(params, time_limit=10.0, seed=42):
    """L-BFGS-B optimization with penalty method."""
    best_p = params.reshape(N, 3).copy()
    best_score = _validate_and_score(best_p.flatten())[1]
    start = time.time()

    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3*i] = 0.0
        hi[3*i] = 1.0
        lo[3*i+1] = 0.0
        hi[3*i+1] = 1.0
        lo[3*i+2] = 1e-6
        hi[3*i+2] = 0.5
    bounds = list(zip(lo, hi))

    try:
        res = minimize(
            _objective,
            best_p.flatten(),
            method='L-BFGS-B',
            jac=_objective_grad,
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        p_new = res.x.reshape(N, 3)
        p_new = _fix_config(p_new.flatten()).reshape(N, 3)
        score = _validate_and_score(p_new.flatten())[1]
        if score > best_score:
            best_score = score
            best_p = p_new
    except Exception:
        pass

    return best_p


def _delaunay_polish(params, time_limit=15.0, seed=42):
    """Polish using Delaunay contact graph for efficiency."""
    best_p = params.reshape(N, 3).copy()
    best_score = _validate_and_score(best_p.flatten())[1]
    start = time.time()

    for iteration in range(5):
        if time.time() - start > time_limit:
            break

        # Get current Delaunay edges
        edges = _get_delaunay_edges(best_p)

        lo = np.zeros(3 * N)
        hi = np.zeros(3 * N)
        for i in range(N):
            lo[3*i] = 0.0
            hi[3*i] = 1.0
            lo[3*i+1] = 0.0
            hi[3*i+1] = 1.0
            lo[3*i+2] = 1e-6
            hi[3*i+2] = 0.5
        bounds = list(zip(lo, hi))

        try:
            res = minimize(
                lambda x: _delaunay_objective(x, edges),
                best_p.flatten(),
                method='L-BFGS-B',
                jac=lambda x: _delaunay_objective_grad(x, edges),
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            p_new = res.x.reshape(N, 3)
            p_new = _fix_config(p_new.flatten()).reshape(N, 3)
            score = _validate_and_score(p_new.flatten())[1]
            if score > best_score:
                best_score = score
                best_p = p_new
        except Exception:
            pass

    return best_p


def _simulated_annealing(params, time_limit=15.0, seed=42):
    """Simulated annealing for global exploration."""
    best_p = params.reshape(N, 3).copy()
    best_score = _validate_and_score(best_p.flatten())[1]
    current_p = best_p.copy()
    current_score = best_score
    rng = np.random.RandomState(seed)
    start = time.time()

    T = 0.05
    T_min = 0.001
    cooling = 0.995
    max_iter = 10000

    for it in range(max_iter):
        if time.time() - start > time_limit:
            break

        # Propose move
        p_new = current_p.copy()
        idx = rng.randint(N)
        p_new[idx, 0] += rng.normal(0, 0.02)
        p_new[idx, 1] += rng.normal(0, 0.02)
        p_new[idx, 2] += rng.normal(0, 0.01)
        p_new = np.clip(p_new, 0, 1)
        p_new[:, 2] = np.clip(p_new[:, 2], 1e-6, 0.5)

        # Fix
        p_new = _fix_config(p_new.flatten()).reshape(N, 3)
        score_new = _validate_and_score(p_new.flatten())[1]

        # Acceptance
        delta = score_new - current_score
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_p = p_new
            current_score = score_new
            if score_new > best_score:
                best_score = score_new
                best_p = p_new.copy()

        T = max(T * cooling, T_min)

    return best_p


def _refine_radii(params, time_limit=5.0):
    """Try to increase radii of circles that have room."""
    best_p = params.reshape(N, 3).copy()
    best_score = _validate_and_score(best_p.flatten())[1]
    start = time.time()

    for _ in range(100):
        if time.time() - start > time_limit:
            break

        p = best_p.copy()
        improved = False

        for i in range(N):
            # Check if we can increase radius
            max_r = min(p[i, 0], 1 - p[i, 0], p[i, 1], 1 - p[i, 1])
            if p[i, 2] < max_r - 1e-6:
                # Try increasing slightly
                new_r = p[i, 2] + 0.001
                p[i, 2] = new_r

                # Check if still valid
                valid, score = _validate_and_score(p.flatten())
                if valid and score > best_score:
                    best_score = score
                    best_p = p.copy()
                    improved = True
                else:
                    p[i, 2] -= 0.001

        if not improved:
            break

    return best_p


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_params = None
    best_score = -np.inf

    initializers = [
        lambda s: _initial_grid(s),
        lambda s: _initial_hex(s),
        lambda s: _initial_corner_weighted(s),
        lambda s: _initial_random(s),
    ]

    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678,
             11, 22, 33, 44, 55, 66, 77, 88, 99, 100,
             200, 300, 400, 500, 600, 700, 800, 900, 1000, 1111]

    # Phase 1: Multi-start with force optimization
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        for init_fn in initializers:
            if time.time() > deadline - 30:
                break

            x0 = init_fn(seed)
            x0 = _fix_config(x0).flatten()

            try:
                p_force = _force_optimize(x0.reshape(N, 3), time_limit=3.0, seed=seed)
                s_force = _validate_and_score(p_force.flatten())[1]

                if s_force > best_score:
                    best_score = s_force
                    best_params = p_force.copy()

            except Exception:
                continue

    if best_params is None:
        best_params = _fix_config(_initial_grid(42)).reshape(N, 3)
        best_score = _validate_and_score(best_params.flatten())[1]

    # Phase 2: L-BFGS refinement
    remaining = deadline - time.time() - 20
    if remaining > 5:
        for s_seed in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 20:
                break
            t_limit = min(4.0, remaining / 5)
            try:
                p_lbfgs = _lbfgs_optimize(best_params, time_limit=t_limit, seed=s_seed)
                s_lbfgs = _validate_and_score(p_lbfgs.flatten())[1]
                if s_lbfgs > best_score:
                    best_score = s_lbfgs
                    best_params = p_lbfgs
            except Exception:
                continue

    # Phase 3: Simulated annealing for global exploration
    remaining = deadline - time.time() - 15
    if remaining > 5:
        for s_seed in [10, 11, 12, 13, 14]:
            if time.time() > deadline - 15:
                break
            t_limit = min(3.0, remaining / 5)
            try:
                p_sa = _simulated_annealing(best_params, time_limit=t_limit, seed=s_seed)
                s_sa = _validate_and_score(p_sa.flatten())[1]
                if s_sa > best_score:
                    best_score = s_sa
                    best_params = p_sa
            except Exception:
                continue

    # Phase 4: Delaunay polish
    remaining = deadline - time.time() - 10
    if remaining > 3:
        try:
            p_polished = _delaunay_polish(best_params, time_limit=min(8.0, remaining), seed=999)
            s_polished = _validate_and_score(p_polished.flatten())[1]
            if s_polished > best_score:
                best_score = s_polished
                best_params = p_polished
        except Exception:
            pass

    # Phase 5: Final radius refinement
    remaining = deadline - time.time() - 3
    if remaining > 2:
        try:
            p_refined = _refine_radii(best_params, time_limit=min(3.0, remaining))
            s_refined = _validate_and_score(p_refined.flatten())[1]
            if s_refined > best_score:
                best_score = s_refined
                best_params = p_refined
        except Exception:
            pass

    # Final safety: enforce validity
    best_params = _fix_config(best_params.flatten()).reshape(N, 3)
    return best_params
