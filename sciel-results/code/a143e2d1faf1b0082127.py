"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, cKDTree

N = 26
EPS = 1e-6


def _build_neighbor_graph(params, r=0.35):
    """Build a sparse neighbor graph using cKDTree for the current configuration."""
    p = params.reshape(N, 3)
    centers = p[:, :2]
    tree = cKDTree(centers)
    pairs = list(tree.query_pairs(r=r))
    if len(pairs) < N:
        try:
            tri = Delaunay(centers)
            for simplex in tri.simplices:
                for a in range(3):
                    for b in range(a + 1, 3):
                        i, j = int(simplex[a]), int(simplex[b])
                        if (i, j) not in pairs and (j, i) not in pairs:
                            pairs.append((i, j))
        except Exception:
            pass
    return pairs


def _objective(params, neighbor_pairs=None, penalty=200.0):
    """Penalty objective focused on neighbor pairs for efficiency."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    r_eff = np.maximum(r, 1e-4)
    score = -np.sum(r_eff)

    wx = np.maximum(0.0, r_eff - x) + np.maximum(0.0, r_eff - (1.0 - x))
    wy = np.maximum(0.0, r_eff - y) + np.maximum(0.0, r_eff - (1.0 - y))
    wall = np.sum(wx) + np.sum(wy)

    if neighbor_pairs is not None and len(neighbor_pairs) > 0:
        iu = np.array(neighbor_pairs).T
    else:
        iu = np.triu_indices(N, k=1)

    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap = np.maximum(0.0, (r_eff[iu[0]] + r_eff[iu[1]]) - dist)
    pair = np.sum(overlap)

    return score + penalty * (pair + wall)


def _objective_grad(params, neighbor_pairs=None, penalty=200.0):
    """Gradient of _objective with neighbor graph support."""
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    r_eff = np.maximum(r, 1e-4)
    n = N

    g = np.zeros_like(params)
    g[2::3] = -1.0

    for i in range(n):
        if r_eff[i] - x[i] > 0:
            g[3 * i + 0] += penalty * (-1.0)
            g[3 * i + 2] += penalty * (+1.0)
        if r_eff[i] - (1.0 - x[i]) > 0:
            g[3 * i + 0] += penalty * (+1.0)
            g[3 * i + 2] += penalty * (+1.0)
        if r_eff[i] - y[i] > 0:
            g[3 * i + 1] += penalty * (-1.0)
            g[3 * i + 2] += penalty * (+1.0)
        if r_eff[i] - (1.0 - y[i]) > 0:
            g[3 * i + 1] += penalty * (+1.0)
            g[3 * i + 2] += penalty * (+1.0)

    if neighbor_pairs is not None and len(neighbor_pairs) > 0:
        iu = np.array(neighbor_pairs).T
    else:
        iu = np.triu_indices(n, k=1)

    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx * dx + dy * dy)
    dist_safe = np.maximum(dist, 1e-8)
    ov = (r_eff[iu[0]] + r_eff[iu[1]]) - dist
    active = ov > 0
    if np.any(active):
        ax = iu[0][active]
        ay = iu[1][active]
        adx = dx[active] / dist_safe[active]
        ady = dy[active] / dist_safe[active]
        for k in range(len(ax)):
            i = ax[k]; j = ay[k]
            g[3 * i + 0] += penalty * (-adx[k])
            g[3 * j + 0] += penalty * (+adx[k])
            g[3 * i + 1] += penalty * (-ady[k])
            g[3 * j + 1] += penalty * (+ady[k])
            g[3 * i + 2] += penalty * (+1.0)
            g[3 * j + 2] += penalty * (+1.0)

    return g


def _physics_relax(params, iters=400, deadline=np.inf):
    """Physics-based relaxation: push apart overlapping circles, clamp to walls,
    then grow each circle to its max feasible radius. Returns improved params."""
    p = params.reshape(N, 3).copy()
    for _ in range(iters):
        if time.time() > deadline:
            break
        # Push apart overlapping pairs
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        need = p[iu[0], 2] + p[iu[1], 2] - dist
        active = need > 1e-9
        if np.any(active):
            for k in range(len(active)):
                i = iu[0][k]; j = iu[1][k]
                d = dist[k]
                if d < 1e-9:
                    p[i, 0] += 1e-3
                    p[j, 0] -= 1e-3
                    continue
                ux = (p[i, 0] - p[j, 0]) / d
                uy = (p[i, 1] - p[j, 1]) / d
                push = need[k] * 0.5
                p[i, 0] += ux * push
                p[i, 1] += uy * push
                p[j, 0] -= ux * push
                p[j, 1] -= uy * push
        # Clamp centers into walls (allow slight radius)
        for i in range(N):
            r = p[i, 2]
            p[i, 0] = np.clip(p[i, 0], r, 1.0 - r)
            p[i, 1] = np.clip(p[i, 1], r, 1.0 - r)
        # Grow each circle to max feasible radius
        for i in range(N):
            max_r = min(p[i, 0], 1.0 - p[i, 0], p[i, 1], 1.0 - p[i, 1])
            for j in range(N):
                if i == j:
                    continue
                d = np.sqrt((p[i, 0] - p[j, 0]) ** 2 + (p[i, 1] - p[j, 1]) ** 2)
                max_r = min(max_r, d - p[j, 2])
            p[i, 2] = max(max_r, 1e-4)
    return p.reshape(-1)


def _grid_initial(seed):
    """Grid of equal circles, first N used."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
    cells = cells[:N]
    arr = np.array(cells, dtype=float).reshape(-1)
    arr += rng.uniform(-1e-4, 1e-4, size=arr.size)
    return arr


def _random_initial(seed):
    """Random feasible-ish positions with modest radii."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, size=N)
    y = rng.uniform(0.1, 0.9, size=N)
    r = np.full(N, 0.1)
    params = np.empty(3 * N)
    params[0::3] = x
    params[1::3] = y
    params[2::3] = r
    return params


def _bounds():
    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3 * i + 0] = 0.0
        hi[3 * i + 0] = 1.0
        lo[3 * i + 1] = 0.0
        hi[3 * i + 1] = 1.0
        lo[3 * i + 2] = 1e-4
        hi[3 * i + 2] = 0.5
    return (lo, hi)


def _refine(params, deadline, rng, budget_per_opt):
    """Neighbor-graph-aware refinement with physics pre-relaxation."""
    lo, hi = _bounds()
    best = params.copy()
    best_val = _objective(params)

    # Phase 0: Physics pre-relaxation (grow circles early)
    phys = _physics_relax(params, iters=200, deadline=deadline)
    if _objective(phys) < best_val:
        best_val = _objective(phys)
        best = phys.copy()
    params = best.copy()

    # Phase 1: Initial L-BFGS-B with all pairs
    for penalty in [200.0, 500.0]:
        if time.time() > deadline:
            break
        def obj_all(p, pen=penalty):
            return _objective(p, neighbor_pairs=None, penalty=pen)
        def grad_all(p, pen=penalty):
            return _objective_grad(p, neighbor_pairs=None, penalty=pen)

        res = minimize(obj_all, params, jac=grad_all, method='L-BFGS-B',
                       bounds=list(zip(lo, hi)),
                       options={'maxiter': budget_per_opt, 'maxfun': budget_per_opt})
        params = res.x
        val = _objective(params)
        if val < best_val:
            best_val = val
            best = params.copy()

    # Phase 2: Contact-graph-aware iterative refinement with physics polish
    for round_num in range(8):
        if time.time() > deadline:
            break
        params = best.copy()
        neighbors = _build_neighbor_graph(params)

        for penalty in [800.0, 1500.0]:
            if time.time() > deadline:
                break
            def obj_ng(p, pen=penalty, ng=neighbors):
                return _objective(p, neighbor_pairs=ng, penalty=pen)
            def grad_ng(p, pen=penalty, ng=neighbors):
                return _objective_grad(p, neighbor_pairs=ng, penalty=pen)

            res = minimize(obj_ng, params, jac=grad_ng, method='L-BFGS-B',
                           bounds=list(zip(lo, hi)),
                           options={'maxiter': max(50, budget_per_opt // 4),
                                    'maxfun': max(50, budget_per_opt // 4)})
            params = res.x
            val = _objective(params, neighbor_pairs=None)
            if val < best_val:
                best_val = val
                best = params.copy()
                if round_num < 7:
                    neighbors = _build_neighbor_graph(best)

        # Physics polish after each round (grow circles to fill space)
        if time.time() < deadline:
            phys = _physics_relax(best, iters=60, deadline=deadline)
            pval = _objective(phys, neighbor_pairs=None)
            if pval < best_val:
                best_val = pval
                best = phys.copy()

    # Phase 3: Simulated annealing with neighbor-aware moves
    params = best.copy()
    val = _objective(params)
    T = 0.04
    step = 0.015
    iters = 3000
    for it in range(iters):
        if time.time() > deadline:
            break
        i = rng.randint(N)
        cand = params.copy()
        cand[3 * i + 0] = params[3 * i + 0] + rng.uniform(-step, step)
        cand[3 * i + 1] = params[3 * i + 1] + rng.uniform(-step, step)
        cand[3 * i + 2] = params[3 * i + 2] + rng.uniform(-step * 0.4, step * 0.4)
        cand[3 * i + 0] = np.clip(cand[3 * i + 0], 0.0, 1.0)
        cand[3 * i + 1] = np.clip(cand[3 * i + 1], 0.0, 1.0)
        cand[3 * i + 2] = np.clip(cand[3 * i + 2], 1e-4, 0.5)
        new_val = _objective(cand)
        d = new_val - val
        if d < 0 or rng.rand() < np.exp(-d / max(T, 1e-6)):
            params = cand
            val = new_val
            if val < best_val:
                best_val = val
                best = params.copy()
        T *= 0.9992
        step *= 0.9997

    # Phase 4: Final polish with full gradient + physics
    if time.time() < deadline:
        res = minimize(lambda p: _objective(p, neighbor_pairs=None),
                       best,
                       jac=lambda p: _objective_grad(p, neighbor_pairs=None),
                       method='L-BFGS-B',
                       bounds=list(zip(lo, hi)),
                       options={'maxiter': 300, 'maxfun': 300})
        if _objective(res.x, neighbor_pairs=None) < best_val:
            best = res.x
            best_val = _objective(best, neighbor_pairs=None)
        if time.time() < deadline:
            phys = _physics_relax(best, iters=80, deadline=deadline)
            pval = _objective(phys, neighbor_pairs=None)
            if pval < best_val:
                best = phys
                best_val = pval

    return best, best_val


def _score_and_clamp(params):
    """Return a valid (x,y,r) array: enforce boundaries and non-overlap by shrinking."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.clip(p[:, 2], 1e-6, np.maximum(wall, 1e-6))

    for _ in range(200):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        need = p[iu[0], 2] + p[iu[1], 2] - dist
        active = need > EPS
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            gap = need[k]
            tot = p[i, 2] + p[j, 2]
            if tot <= EPS:
                continue
            shrink_i = min(p[i, 2] - 1e-6, gap * 0.5)
            shrink_j = min(p[j, 2] - 1e-6, gap - shrink_i)
            if shrink_i < 0:
                shrink_i = 0.0
            if shrink_j < 0:
                shrink_j = 0.0
            p[i, 2] -= shrink_i
            p[j, 2] -= shrink_j
            p[i, 2] = min(p[i, 2], np.maximum(np.minimum(np.minimum(p[i, 0], 1 - p[i, 0]),
                                                         np.minimum(p[i, 1], 1 - p[i, 1])), 1e-6))
            p[j, 2] = min(p[j, 2], np.maximum(np.minimum(np.minimum(p[j, 0], 1 - p[j, 0]),
                                                         np.minimum(p[j, 1], 1 - p[j, 1])), 1e-6))
    return p


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    rng = np.random.RandomState(12345)

    inits = []
    inits.append(_grid_initial(0))
    for s in range(1, 8):
        inits.append(_random_initial(s))

    best_params = None
    best_val = np.inf
    best_score = 0.0

    for idx, init in enumerate(inits):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        n_left = len(inits) - idx
        per_init = max(1.0, remaining / max(n_left, 1))
        budget_per_opt = int(min(3000, max(400, per_init * 1200)))
        params, val = _refine(init, deadline, rng, budget_per_opt)
        feas = _score_and_clamp(params)
        sc = float(np.sum(feas[:, 2]))
        if sc > best_score:
            best_score = sc
            best_params = feas

    if best_params is None:
        grid = _grid_initial(0)
        best_params = _score_and_clamp(grid)
        best_score = float(np.sum(best_params[:, 2]))

    return best_params
