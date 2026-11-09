"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)
_n_triangles = len(_idx_i)

# Precompute all C(11,2) pair index pairs
_pair_i, _pair_j = [], []
for i in range(N):
    for j in range(i + 1, N):
        _pair_i.append(i)
        _pair_j.append(j)
_pair_i = np.array(_pair_i)
_pair_j = np.array(_pair_j)
_n_pairs = len(_pair_i)


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas.min()


def _min_pairwise_dist(points):
    """Compute minimum pairwise distance among all C(N,2) pairs."""
    p1 = points[_pair_i]
    p2 = points[_pair_j]
    d = np.sqrt(np.sum((p2 - p1) ** 2, axis=1))
    return d.min()


def _is_inside(p):
    """Check if point p=(x,y) is inside the equilateral triangle."""
    x, y = p
    if y < -1e-9 or x < -1e-9 or x > 1 + 1e-9:
        return False
    if y > np.sqrt(3) * x + 1e-9:
        return False
    if y > np.sqrt(3) * (1 - x) + 1e-9:
        return False
    return True


def _bary_to_cart(bary):
    """Convert barycentric coords (u,v) with u+v<=1 to Cartesian.
    Point = u*A + v*B + (1-u-v)*C where A=(0,0), B=(1,0), C=(0.5,H)."""
    u = bary[0]
    v = bary[1]
    w = 1.0 - u - v
    x = u * 0.0 + v * 1.0 + w * 0.5
    y = u * 0.0 + v * 0.0 + w * H
    return np.array([x, y])


def _params_to_points(params):
    """Convert 2*N params (barycentric u,v pairs) to N Cartesian points.
    Each pair (u,v) must satisfy u>=0, v>=0, u+v<=1."""
    points = np.zeros((N, 2))
    for i in range(N):
        u = params[2 * i]
        v = params[2 * i + 1]
        if u < 0: u = 0.0
        if v < 0: v = 0.0
        if u + v > 1.0:
            s = 1.0 / (u + v)
            u *= s
            v *= s
        points[i] = np.array([v + (1 - u - v) * 0.5, (1 - u - v) * H])
    return points


def _soft_min_area(params, beta=60.0):
    """Compute soft-min of triangle areas via log-sum-exp for a smooth, differentiable objective.
    Minimizing this maximizes the minimum area in a differentiable way.
    Returns (value, gradient) for L-BFGS-B."""
    points = _params_to_points(params)
    # Penalty if any point is outside
    for i in range(N):
        if not _is_inside(points[i]):
            return 1e6, np.zeros(2 * N)

    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)

    # Soft min via log-sum-exp: softmin(a) = -1/beta * log(sum(exp(-beta*a)))
    # We want to maximize softmin, so minimize -softmin = 1/beta * log(sum(exp(-beta*a)))
    # For numerical stability, subtract max
    neg_beta_areas = -beta * areas
    m = np.max(neg_beta_areas)
    log_sum = m + np.log(np.sum(np.exp(neg_beta_areas - m)))
    value = log_sum / beta

    # Gradient via finite differences (robust, avoids complex chain rule through |cross|)
    eps = 1e-6
    grad = np.zeros(2 * N)
    for d in range(2 * N):
        params_plus = params.copy()
        params_plus[d] += eps
        val_plus = _soft_min_area_value(params_plus, beta)
        params_minus = params.copy()
        params_minus[d] -= eps
        val_minus = _soft_min_area_value(params_minus, beta)
        grad[d] = (val_plus - val_minus) / (2 * eps)

    return value, grad


def _soft_min_area_value(params, beta=60.0):
    """Just the value part of soft min area (for finite-diff gradient)."""
    points = _params_to_points(params)
    for i in range(N):
        if not _is_inside(points[i]):
            return 1e6
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    neg_beta_areas = -beta * areas
    m = np.max(neg_beta_areas)
    log_sum = m + np.log(np.sum(np.exp(neg_beta_areas - m)))
    return log_sum / beta


def _objective_exact(params):
    """Minimize negative exact min-area."""
    points = _params_to_points(params)
    for i in range(N):
        if not _is_inside(points[i]):
            return 1e6
    return -_min_area(points)


def _objective_packing(params):
    """Minimize negative min pairwise distance (smooth packing proxy)."""
    points = _params_to_points(params)
    for i in range(N):
        if not _is_inside(points[i]):
            return 1e6
    return -_min_pairwise_dist(points)


def _lattice_initial_points():
    """Generate initial points on a triangular lattice inside the equilateral triangle.
    Uses barycentric coordinates on a grid."""
    points = []
    for row in range(4):
        for col in range(4 - row):
            u = row / 3.0
            v = col / 3.0
            if u + v <= 1.0 + 1e-9:
                pt = _bary_to_cart(np.array([u, v]))
                cx, cy = 0.5, H / 3
                pt = 0.98 * pt + 0.02 * np.array([cx, cy])
                points.append(pt)
    points.append(np.array([0.5, H / 3]))
    return np.array(points[:N])


def _lattice_params(points):
    """Convert Cartesian points to barycentric params (u,v) for each point."""
    params = np.zeros(2 * N)
    for i in range(N):
        x, y = points[i]
        s = 1.0 - y / H  # u+v
        d = 2.0 * (x - 0.5)  # v-u
        u = (s - d) / 2.0
        v = (s + d) / 2.0
        params[2*i] = max(0.0, u)
        params[2*i+1] = max(0.0, v)
    return params


def _row_guess():
    """Row-based symmetric initial guess (5,3,2,1)."""
    y_levels = [0.08 * H, 0.35 * H, 0.62 * H, 0.88 * H]
    counts = [5, 3, 2, 1]
    pts = []
    for y, c in zip(y_levels, counts):
        x_min = y / np.sqrt(3)
        x_max = 1.0 - y / np.sqrt(3)
        if c > 1:
            margin = 0.03 * (x_max - x_min)
            xs = np.linspace(x_min + margin, x_max - margin, c)
        else:
            xs = np.array([(x_min + x_max) / 2])
        for x in xs:
            pts.append([x, y])
    return np.array(pts)


def construct():
    t_start = time.time()
    deadline = t_start + 140.0

    best_params = None
    best_score = -np.inf

    def _try_update(params, score):
        nonlocal best_params, best_score
        if score > best_score:
            best_params = params.copy()
            best_score = score

    # Phase 1: Lattice-based initial configuration + packing optimization
    try:
        lat_pts = _lattice_initial_points()
        lat_params = _lattice_params(lat_pts)
        res_pack = minimize(
            _objective_packing,
            lat_params,
            method='Nelder-Mead',
            options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        _try_update(res_pack.x, -_objective_exact(res_pack.x))
    except Exception:
        pass

    # Phase 2: Differential evolution on packing objective for global search
    if time.time() < deadline - 40.0:
        try:
            bounds = [(0.0, 1.0)] * (2 * N)
            result = differential_evolution(
                _objective_packing,
                bounds=bounds,
                maxiter=80,
                popsize=25,
                tol=1e-6,
                seed=42,
                polish=False,
                updating='immediate',
                init='sobol'
            )
            _try_update(result.x, -_objective_exact(result.x))
        except Exception:
            pass

    # Phase 3: NEW - Gradient-based L-BFGS-B refinement on the differentiable soft-min area
    # The soft-min (log-sum-exp) turns the nonsmooth max-min into a smooth landscape so
    # gradient methods can navigate the non-convex topology and climb toward the true min.
    if best_params is not None and time.time() < deadline - 30.0:
        try:
            for beta in [40.0, 80.0, 120.0]:
                if time.time() > deadline - 30.0:
                    break
                res_soft = minimize(
                    lambda p: _soft_min_area_value(p, beta),
                    best_params,
                    jac=lambda p: _soft_min_area_grad(p, beta),
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * (2 * N),
                    options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 200000}
                )
                _try_update(res_soft.x, -_objective_exact(res_soft.x))
        except Exception:
            pass

    # Phase 4: Polish with exact min-area objective from best packing solution
    if time.time() < deadline - 20.0:
        try:
            res3 = minimize(
                _objective_exact,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            _try_update(res3.x, -res3.fun)
        except Exception:
            pass

    # Phase 5: Multiple restarts - mix of lattice perturbations, row guess, random; each with
    # a gradient (soft-min) pass followed by an exact polish.
    if time.time() < deadline - 15.0:
        rng = np.random.RandomState(777)
        for restart in range(10):
            if time.time() > deadline - 15.0:
                break
            try:
                if restart < 4:
                    lat_pts = _lattice_initial_points()
                    noise = rng.randn(N, 2) * 0.08
                    lat_pts = lat_pts + noise
                    for i in range(N):
                        if not _is_inside(lat_pts[i]):
                            lat_pts[i] = 0.9 * lat_pts[i] + 0.1 * np.array([0.5, H/3])
                    params = _lattice_params(lat_pts)
                elif restart < 6:
                    row_pts = _row_guess()
                    for i in range(N):
                        if not _is_inside(row_pts[i]):
                            row_pts[i] = 0.9 * row_pts[i] + 0.1 * np.array([0.5, H/3])
                    params = _lattice_params(row_pts)
                else:
                    params = np.zeros(2 * N)
                    for i in range(N):
                        u, v = rng.random(), rng.random()
                        if u + v > 1.0:
                            s = 1.0 / (u + v)
                            u *= s
                            v *= s
                        params[2*i] = u
                        params[2*i+1] = v

                # Gradient pass on soft-min (smooth surrogate)
                if time.time() < deadline - 15.0:
                    res_g = minimize(
                        lambda p: _soft_min_area_value(p, 60.0),
                        params,
                        jac=lambda p: _soft_min_area_grad(p, 60.0),
                        method='L-BFGS-B',
                        bounds=[(0.0, 1.0)] * (2 * N),
                        options={'maxiter': 1500, 'ftol': 1e-14, 'maxfun': 100000}
                    )
                    params = res_g.x
                # Then exact polish
                res_e = minimize(
                    _objective_exact,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                _try_update(res_e.x, -res_e.fun)
            except Exception:
                pass

    # Phase 6: Final fine polish (exact + one more gradient pass for high precision)
    if time.time() < deadline - 5.0:
        try:
            res_final = minimize(
                _objective_exact,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 30000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            _try_update(res_final.x, -res_final.fun)
            if time.time() < deadline - 3.0:
                res_g2 = minimize(
                    lambda p: _soft_min_area_value(p, 150.0),
                    best_params,
                    jac=lambda p: _soft_min_area_grad(p, 150.0),
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * (2 * N),
                    options={'maxiter': 3000, 'ftol': 1e-15, 'maxfun': 200000}
                )
                _try_update(res_g2.x, -_objective_exact(res_g2.x))
        except Exception:
            pass

    if best_params is not None:
        points = _params_to_points(best_params)
        for i in range(N):
            if not _is_inside(points[i]):
                points[i] = 0.95 * points[i] + 0.05 * np.array([0.5, H / 3])
        return points
    else:
        rng = np.random.RandomState(0)
        pts = []
        for _ in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 0.99:
                    pts.append([u + 0.5 * v, H * v])
                    break
        return np.array(pts)


def _soft_min_area_grad(params, beta=60.0):
    """Finite-difference gradient of the soft-min area value (used by L-BFGS-B)."""
    eps = 1e-6
    grad = np.zeros(2 * N)
    for d in range(2 * N):
        p_plus = params.copy()
        p_plus[d] += eps
        p_minus = params.copy()
        p_minus[d] -= eps
        grad[d] = (_soft_min_area_value(p_plus, beta) - _soft_min_area_value(p_minus, beta)) / (2 * eps)
    return grad

# EVOLVE-BLOCK-END
