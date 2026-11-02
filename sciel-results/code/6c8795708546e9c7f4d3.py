"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
DEADLINE = 140.0  # safety margin before 150s kill


def _contact_pairs(params, margin=0.0):
    """
    Find pairs of circles that are close (within r_i + r_j + margin).
    Returns array of (i, j) pairs.
    """
    p = params.reshape(N, 3)
    centers = p[:, :2]
    radii = p[:, 2]

    iu = np.triu_indices(N, k=1)
    dx = centers[iu[0], 0] - centers[iu[1], 0]
    dy = centers[iu[0], 1] - centers[iu[1], 1]
    dist = np.sqrt(dx * dx + dy * dy)
    threshold = radii[iu[0]] + radii[iu[1]] + margin
    active = dist < threshold
    return np.column_stack([iu[0][active], iu[1][active]])


def _penalty_objective(params, pairs, penalty_weight=100.0, wall_weight=100.0):
    """
    Penalty-based objective: maximize sum(radii) - penalty for overlaps.
    Uses only contact pairs for sparsity.
    """
    p = params.reshape(N, 3)
    centers = p[:, :2]
    radii = np.maximum(p[:, 2], 1e-8)

    # Objective: maximize sum of radii
    obj = -np.sum(radii)

    # Overlap penalty for contact pairs
    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        dx = centers[i_idx, 0] - centers[j_idx, 0]
        dy = centers[i_idx, 1] - centers[j_idx, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = np.maximum(radii[i_idx] + radii[j_idx] - dist, 0.0)
        obj += penalty_weight * np.sum(overlap ** 2)

    # Wall violation penalty
    wall_viol = np.maximum(
        np.minimum(radii - centers[:, 0],
                   np.minimum(radii - (1.0 - centers[:, 0]),
                              np.minimum(radii - centers[:, 1],
                                         radii - (1.0 - centers[:, 1])))),
        0.0)
    obj += wall_weight * np.sum(wall_viol ** 2)

    return obj


def _penalty_gradient(params, pairs, penalty_weight=100.0, wall_weight=100.0):
    """Gradient of penalty objective."""
    p = params.reshape(N, 3)
    centers = p[:, :2]
    radii = np.maximum(p[:, 2], 1e-8)
    grad = np.zeros(3 * N)

    # d(-sum(radii))/dr_i = -1
    grad[2::3] = -1.0

    # Overlap penalty gradient
    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        dx = centers[i_idx, 0] - centers[j_idx, 0]
        dy = centers[i_idx, 1] - centers[j_idx, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = np.maximum(dist, 1e-10)
        overlap = np.maximum(radii[i_idx] + radii[j_idx] - dist, 0.0)
        active = overlap > 0

        if np.any(active):
            ux = dx[active] / dist_safe[active]
            uy = dy[active] / dist_safe[active]
            ov = overlap[active]
            ii = i_idx[active]
            jj = j_idx[active]

            # d/dx_i = 2 * w * ov * (dx/dist)
            grad[ii * 3] += 2.0 * penalty_weight * ov * ux
            grad[ii * 3 + 1] += 2.0 * penalty_weight * ov * uy
            # d/dr_i = 2 * w * ov * (-1)  [because overlap = r_i + r_j - dist]
            grad[ii * 3 + 2] += -2.0 * penalty_weight * ov

            grad[jj * 3] += -2.0 * penalty_weight * ov * ux
            grad[jj * 3 + 1] += -2.0 * penalty_weight * ov * uy
            grad[jj * 3 + 2] += -2.0 * penalty_weight * ov

    # Wall penalty gradient
    xi = centers[:, 0]
    yi = centers[:, 1]
    r = radii

    # For each circle, compute which wall is violated
    w_left = np.maximum(r - xi, 0.0)
    w_right = np.maximum(r - (1.0 - xi), 0.0)
    w_bottom = np.maximum(r - yi, 0.0)
    w_top = np.maximum(r - (1.0 - yi), 0.0)

    # d/dx: -2*w*wl for left, +2*w*wr for right
    grad[0::3] += -2.0 * wall_weight * w_left + 2.0 * wall_weight * w_right
    # d/dy: -2*w*wb for bottom, +2*w*wt for top
    grad[1::3] += -2.0 * wall_weight * w_bottom + 2.0 * wall_weight * w_top
    # d/dr: +2*w*(wl+wr+wb+wt)
    grad[2::3] += 2.0 * wall_weight * (w_left + w_right + w_bottom + w_top)

    return grad


def _lbfgs_with_contacts(params, bounds, penalty_weight=50.0, maxiter=200):
    """
    L-BFGS-B optimization with dynamic contact graph rebuilding.
    """
    p = params.copy()
    best_p = p.copy()
    best_score = -np.inf

    # Compute initial valid score
    p_valid = _enforce_validity(p)
    best_score = _score_valid(p_valid)
    best_p = p_valid.copy()

    for outer in range(5):
        # Rebuild contact pairs
        p_valid = _enforce_validity(p)
        pairs = _contact_pairs(p_valid, margin=0.02)

        if len(pairs) == 0:
            pairs = np.triu_indices(N, k=1)

        def obj(x):
            return _penalty_objective(x, pairs, penalty_weight, 100.0)

        def grad(x):
            return _penalty_gradient(x, pairs, penalty_weight, 100.0)

        try:
            res = minimize(obj, p, jac=grad, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-12})

            if hasattr(res, 'x'):
                res_valid = _enforce_validity(res.x)
                s = _score_valid(res_valid)
                if s > best_score + 1e-10:
                    best_score = s
                    best_p = res_valid.copy()
                    p = res.x.copy()
                else:
                    break
        except Exception:
            break

    return best_p


def _enforce_validity(params):
    """Enforce all constraints: walls and pairwise non-overlap by shrinking radii."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    for _ in range(2000):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    return p


def _score_valid(p):
    """Compute sum of radii for a valid configuration."""
    return np.sum(p[:, 2])


def _delaunay_edges(centers):
    """Return Delaunay edges as list of (i, j) pairs."""
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        return list(edges)
    except Exception:
        return []


def _delaunay_slsqp(params, bounds_list, maxiter=300, tol=1e-3):
    """
    SLSQP optimization using Delaunay edges as sparse contact constraints.
    Maximizes sum of radii subject to:
    - dist(i,j) >= r_i + r_j for Delaunay edges
    - wall constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    """
    p = params.reshape(N, 3).copy()
    centers = p[:, :2]
    radii = np.maximum(p[:, 2], 1e-6)

    # Build Delaunay edge list
    edge_list = _delaunay_edges(centers)
    if len(edge_list) == 0:
        return params

    eu = np.array([e[0] for e in edge_list])
    ev = np.array([e[1] for e in edge_list])
    m = len(eu)

    def obj(x):
        return -np.sum(x[2::3])

    def obj_grad(x):
        g = np.zeros_like(x)
        g[2::3] = -1.0
        return g

    def constraint_func(x):
        """dist(i,j) >= r_i + r_j for Delaunay edges."""
        xi = x[eu * 3]
        yi = x[eu * 3 + 1]
        ri = np.maximum(x[eu * 3 + 2], 1e-6)
        xj = x[ev * 3]
        yj = x[ev * 3 + 1]
        rj = np.maximum(x[ev * 3 + 2], 1e-6)
        dx = xi - xj
        dy = yi - yj
        dist = np.sqrt(dx * dx + dy * dy)
        return dist - (ri + rj)

    def constraint_grad(x):
        jacobian = np.zeros((m, 3 * N))
        xi = x[eu * 3]
        yi = x[eu * 3 + 1]
        xj = x[ev * 3]
        yj = x[ev * 3 + 1]
        dx = xi - xj
        dy = yi - yj
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = np.maximum(dist, 1e-10)
        ux = dx / dist_safe
        uy = dy / dist_safe

        for k in range(m):
            i = eu[k]
            j = ev[k]
            jacobian[k, 3 * i] = ux[k]
            jacobian[k, 3 * i + 1] = uy[k]
            jacobian[k, 3 * i + 2] = -1.0
            jacobian[k, 3 * j] = -ux[k]
            jacobian[k, 3 * j + 1] = -uy[k]
            jacobian[k, 3 * j + 2] = -1.0
        return jacobian

    def wall_func(x):
        xi = x[0::3]
        yi = x[1::3]
        ri = np.maximum(x[2::3], 1e-6)
        return np.minimum(np.minimum(xi - ri, 1.0 - xi - ri),
                          np.minimum(yi - ri, 1.0 - yi - ri))

    def wall_grad(x):
        jacobian = np.zeros((N, 3 * N))
        xi = x[0::3]
        yi = x[1::3]
        ri = np.maximum(x[2::3], 1e-6)

        for i in range(N):
            wx_left = xi[i] - ri[i]
            wx_right = 1.0 - xi[i] - ri[i]
            wy_bottom = yi[i] - ri[i]
            wy_top = 1.0 - yi[i] - ri[i]
            wmin = min(wx_left, wx_right, wy_bottom, wy_top)

            if wx_left <= wmin + 1e-12:
                jacobian[i, 3 * i] += 1.0
                jacobian[i, 3 * i + 2] -= 1.0
            if wx_right <= wmin + 1e-12:
                jacobian[i, 3 * i] -= 1.0
                jacobian[i, 3 * i + 2] -= 1.0
            if wy_bottom <= wmin + 1e-12:
                jacobian[i, 3 * i + 1] += 1.0
                jacobian[i, 3 * i + 2] -= 1.0
            if wy_top <= wmin + 1e-12:
                jacobian[i, 3 * i + 1] -= 1.0
                jacobian[i, 3 * i + 2] -= 1.0
        return jacobian

    constraints = [
        {'type': 'ineq', 'fun': constraint_func, 'jac': constraint_grad},
        {'type': 'ineq', 'fun': wall_func, 'jac': wall_grad}
    ]

    try:
        res = minimize(obj, params, jac=obj_grad, method='SLSQP',
                       bounds=bounds_list, constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxfun': 1000})

        if hasattr(res, 'x'):
            res_valid = _enforce_validity(res.x)
            if _score_valid(res_valid) > _score_valid(params.reshape(N, 3)):
                return res.x
    except Exception:
        pass

    return params


def _grow_relax_loop(params, bounds_list, deadline, max_iter=10):
    """
    Iterative grow-and-relax: alternates between L-BFGS-B penalty optimization
    and SLSQP refinement with dynamic contact graph rebuilding.
    """
    p = params.copy()
    best_p = p.copy()
    best_score = _score_valid(p.reshape(N, 3))

    for it in range(max_iter):
        if time.time() > deadline - 5:
            break

        # Phase 1: L-BFGS-B with penalty (fast, global-ish)
        if time.time() < deadline - 10:
            p_lbgfs = _lbfgs_with_contacts(p, bounds_list, penalty_weight=50.0, maxiter=150)
            p_valid = _enforce_validity(p_lbgfs)
            s = _score_valid(p_valid)
            if s > best_score + 1e-10:
                best_score = s
                best_p = p_valid.copy()
                p = p_valid.copy()

        # Phase 2: SLSQP with Delaunay constraints (refinement)
        if time.time() < deadline - 5:
            p_slsqp = _delaunay_slsqp(p, bounds_list, maxiter=150, tol=1e-4)
            p_valid = _enforce_validity(p_slsqp)
            s = _score_valid(p_valid)
            if s > best_score + 1e-10:
                best_score = s
                best_p = p_valid.copy()
                p = p_valid.copy()
            else:
                # Try with lower penalty weight for L-BFGS
                if time.time() < deadline - 3:
                    p_lbgfs2 = _lbfgs_with_contacts(p, bounds_list, penalty_weight=20.0, maxiter=100)
                    p_valid2 = _enforce_validity(p_lbgfs2)
                    s2 = _score_valid(p_valid2)
                    if s2 > best_score + 1e-10:
                        best_score = s2
                        best_p = p_valid2.copy()
                        p = p_valid2.copy()
                    else:
                        break

    return best_p.reshape(-1)


def _hex_initial(seed):
    """Hexagonal grid initialization."""
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

    dists = np.sum((points - 0.5) ** 2, axis=1)
    indices = np.argsort(dists)[:N]

    p = points[indices]
    radii = np.full(N, 0.05)

    params = np.empty(3 * N)
    params[0::3] = p[:, 0]
    params[1::3] = p[:, 1]
    params[2::3] = radii

    params += rng.uniform(-1e-4, 1e-4, size=params.size)
    return params


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


def _voronoi_initial(seed):
    """Initialize using random points with estimated radii."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))

    radii = np.zeros(N)
    for i, p in enumerate(points):
        dists = np.linalg.norm(points - p, axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        wall_dist = np.min([p[0], 1 - p[0], p[1], 1 - p[1]])
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.9

    params = np.empty(3 * N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    params += rng.uniform(-1e-5, 1e-5, size=params.size)
    return params


def _bounds():
    lo = np.zeros(3 * N)
    hi = np.zeros(3 * N)
    for i in range(N):
        lo[3 * i] = 0.0
        hi[3 * i] = 1.0
        lo[3 * i + 1] = 0.0
        hi[3 * i + 1] = 1.0
        lo[3 * i + 2] = 1e-6
        hi[3 * i + 2] = 0.5
    return (lo, hi)


def _anneal_refine(params, bounds_list, deadline):
    """
    Simulated-annealing-inspired perturbation + local refinement.
    Periodically perturbs the configuration and re-optimizes to escape local minima.
    """
    p = params.copy()
    best_p = p.copy()
    best_score = _score_valid(p.reshape(N, 3))
    rng = np.random.RandomState(12345)

    temperature = 0.02
    cooling = 0.85
    max_perturbs = 15

    for k in range(max_perturbs):
        if time.time() > deadline - 8:
            break

        # Perturb positions and radii
        p_perturbed = p.copy()
        p_perturbed[0::3] += rng.normal(0, temperature, N)
        p_perturbed[1::3] += rng.normal(0, temperature, N)
        p_perturbed[2::3] += rng.normal(0, temperature * 0.5, N)

        # Clip to valid range
        p_perturbed[0::3] = np.clip(p_perturbed[0::3], 0.01, 0.99)
        p_perturbed[1::3] = np.clip(p_perturbed[1::3], 0.01, 0.99)
        p_perturbed[2::3] = np.clip(p_perturbed[2::3], 1e-4, 0.45)

        p_valid = _enforce_validity(p_perturbed)
        s = _score_valid(p_valid)

        # Accept if better, or with probability exp(-(s_best - s)/T)
        if s > best_score:
            best_score = s
            best_p = p_valid.copy()
            p = p_valid.copy()
        else:
            delta = best_score - s
            if delta < 1e-10 or rng.random() < np.exp(-delta / max(temperature, 1e-8)):
                p = p_valid.copy()

        # Cool down
        temperature *= cooling

        # Local refinement after perturbation
        if time.time() < deadline - 5:
            p_refined = _lbfgs_with_contacts(p, bounds_list, penalty_weight=30.0, maxiter=80)
            p_refined_valid = _enforce_validity(p_refined)
            s_ref = _score_valid(p_refined_valid)
            if s_ref > best_score:
                best_score = s_ref
                best_p = p_refined_valid.copy()
                p = p_refined_valid.copy()

    return best_p.reshape(-1)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    lo, hi = _bounds()
    bounds_list = list(zip(lo, hi))

    best_params = None
    best_score = -np.inf

    initializers = [
        lambda s: _hex_initial(s),
        lambda s: _grid_initial(s),
        lambda s: _voronoi_initial(s),
    ]

    # Phase 1: Multi-start with L-BFGS-B penalty optimization
    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678,
             11, 22, 33, 44, 55, 66, 77, 88, 99, 100]

    for seed in seeds:
        if time.time() > deadline - 30:
            break
        for init_fn in initializers:
            if time.time() > deadline - 25:
                break

            params = init_fn(seed)
            params_valid = _enforce_validity(params)
            current_score = _score_valid(params_valid)

            if current_score > best_score:
                best_score = current_score
                best_params = params_valid.copy()

            # L-BFGS-B penalty optimization (fast)
            if time.time() < deadline - 15:
                refined = _lbfgs_with_contacts(params_valid.reshape(-1), bounds_list,
                                               penalty_weight=50.0, maxiter=150)
                refined_valid = _enforce_validity(refined)
                refined_score = _score_valid(refined_valid)

                if refined_score > best_score:
                    best_score = refined_score
                    best_params = refined_valid.copy()

    # Phase 2: Grow-and-relax from best state
    if best_params is not None and time.time() < deadline - 20:
        refined = _grow_relax_loop(best_params.reshape(-1), bounds_list, deadline, max_iter=12)
        refined_valid = _enforce_validity(refined)
        refined_score = _score_valid(refined_valid)

        if refined_score > best_score:
            best_score = refined_score
            best_params = refined_valid.copy()

    # Phase 3: Annealing to escape local minima
    if best_params is not None and time.time() < deadline - 10:
        annealed = _anneal_refine(best_params.reshape(-1), bounds_list, deadline)
        annealed_valid = _enforce_validity(annealed)
        annealed_score = _score_valid(annealed_valid)

        if annealed_score > best_score:
            best_score = annealed_score
            best_params = annealed_valid.copy()

    # Phase 4: Final SLSQP polish
    if best_params is not None and time.time() < deadline - 3:
        final = _delaunay_slsqp(best_params, bounds_list, maxiter=200, tol=1e-5)
        final_valid = _enforce_validity(final)
        final_score = _score_valid(final_valid)

        if final_score > best_score:
            best_score = final_score
            best_params = final_valid.copy()

    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_params
