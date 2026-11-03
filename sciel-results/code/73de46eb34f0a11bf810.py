"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import numpy as np
import time
from scipy.spatial import Delaunay, Voronoi
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 145  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0) using log-sum-exp."""
    return np.logaddexp(0, z / eps) * eps


def _validate(circles, tol=1e-7):
    """Check circles are inside unit square and non-overlapping."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(r < -tol) or np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _fix_boundaries(circles):
    """Project circles inside unit square and ensure non-overlap by shrinking if necessary."""
    circles = circles.copy()
    for i in range(N):
        r = max(0.0, circles[i, 2])
        r = min(r, 0.5)
        circles[i, 2] = r
        circles[i, 0] = np.clip(circles[i, 0], r, 1.0 - r)
        circles[i, 1] = np.clip(circles[i, 1], r, 1.0 - r)

    for _iter in range(15):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i, 2] + circles[j, 2]
                if dist < min_dist:
                    overlap = min_dist - dist
                    shrink = overlap / 2.0 + 1e-5
                    circles[i, 2] = max(0.0, circles[i, 2] - shrink)
                    circles[j, 2] = max(0.0, circles[j, 2] - shrink)
                    circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1.0 - circles[i, 2])
                    circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1.0 - circles[i, 2])
                    circles[j, 0] = np.clip(circles[j, 0], circles[j, 2], 1.0 - circles[j, 2])
                    circles[j, 1] = np.clip(circles[j, 1], circles[j, 2], 1.0 - circles[j, 2])
                    changed = True
        if not changed:
            break

    return circles


def _objective_penalty(params, penalty_weight, smooth_eps=1e-4):
    """
    Vectorized objective with soft penalty for constraint violations.
    params: flat array [x0, y0, r0, x1, y1, r1, ...]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    penalty = 0.0

    # Boundary violations (smooth)
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    penalty += np.sum(_softplus(lx, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ux, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ly, smooth_eps) ** 2)
    penalty += np.sum(_softplus(uy, smooth_eps) ** 2)

    # Negative radius
    penalty += np.sum(_softplus(-r, smooth_eps) ** 2)

    # Full pairwise overlaps (vectorized)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlap, smooth_eps) ** 2)

    return -np.sum(r) + penalty_weight * penalty


def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Objective and gradient for penalty method with analytic gradients.
    Returns (value, grad).
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # d(-sum r)/dr

    # Boundary violations
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    # Smooth derivative of softplus(z, eps) is sigmoid(z/eps)
    def _sigmoid(z, eps):
        return 1.0 / (1.0 + np.exp(-z / eps))

    # d/dx of softplus(r-x)^2 = -2*softplus(r-x)*sigmoid((r-x)/eps)
    # d/dr of softplus(r-x)^2 = +2*softplus(r-x)*sigmoid((r-x)/eps)
    # etc.

    # Left x boundary: softplus(r - x, eps)
    sp_lx = _softplus(lx, smooth_eps)
    sg_lx = _sigmoid(lx, smooth_eps)
    grad[0::3] += -2.0 * penalty_weight * sp_lx * sg_lx
    grad[2::3] += 2.0 * penalty_weight * sp_lx * sg_lx

    # Right x boundary: softplus(x + r - 1, eps)
    sp_ux = _softplus(ux, smooth_eps)
    sg_ux = _sigmoid(ux, smooth_eps)
    grad[0::3] += 2.0 * penalty_weight * sp_ux * sg_ux
    grad[2::3] += 2.0 * penalty_weight * sp_ux * sg_ux

    # Left y boundary: softplus(r - y, eps)
    sp_ly = _softplus(ly, smooth_eps)
    sg_ly = _sigmoid(ly, smooth_eps)
    grad[1::3] += -2.0 * penalty_weight * sp_ly * sg_ly
    grad[2::3] += 2.0 * penalty_weight * sp_ly * sg_ly

    # Right y boundary: softplus(y + r - 1, eps)
    sp_uy = _softplus(uy, smooth_eps)
    sg_uy = _sigmoid(uy, smooth_eps)
    grad[1::3] += 2.0 * penalty_weight * sp_uy * sg_uy
    grad[2::3] += 2.0 * penalty_weight * sp_uy * sg_uy

    # Negative radius
    sp_nr = _softplus(-r, smooth_eps)
    sg_nr = _sigmoid(-r, smooth_eps)
    grad[2::3] += -2.0 * penalty_weight * sp_nr * sg_nr

    # Pairwise overlaps
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    sp_ov = _softplus(overlap, smooth_eps)
    sg_ov = _sigmoid(overlap, smooth_eps)

    # d/dx_i of softplus(overlap)^2 = 2*sp*sg * d(overlap)/dx_i = 2*sp*sg * (-dx/dist)
    # d/dx_j of softplus(overlap)^2 = 2*sp*sg * (dx/dist)
    # d/dr_i = 2*sp*sg * 1
    # d/dr_j = 2*sp*sg * 1

    factor = 2.0 * penalty_weight * sp_ov * sg_ov
    inv_dist = 1.0 / dist

    np.add.at(grad[0::3], idx_i, -factor * dx * inv_dist)
    np.add.at(grad[0::3], idx_j, factor * dx * inv_dist)
    np.add.at(grad[1::3], idx_i, -factor * dy * inv_dist)
    np.add.at(grad[1::3], idx_j, factor * dy * inv_dist)
    np.add.at(grad[2::3], idx_i, factor)
    np.add.at(grad[2::3], idx_j, factor)

    val += penalty_weight * (np.sum(sp_lx**2) + np.sum(sp_ux**2) + np.sum(sp_ly**2) + np.sum(sp_uy**2) + np.sum(sp_nr**2) + np.sum(sp_ov**2))

    return val, grad


def _objective_delaunay(params, penalty_weight, smooth_eps=1e-4, contact_edges=None):
    """
    Objective using Delaunay-based contact graph for overlap constraints.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    penalty = 0.0

    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    penalty += np.sum(_softplus(lx, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ux, smooth_eps) ** 2)
    penalty += np.sum(_softplus(ly, smooth_eps) ** 2)
    penalty += np.sum(_softplus(uy, smooth_eps) ** 2)
    penalty += np.sum(_softplus(-r, smooth_eps) ** 2)

    if contact_edges is not None and len(contact_edges) > 0:
        for (i, j) in contact_edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            penalty += _softplus(overlap, smooth_eps) ** 2
    else:
        n = len(r)
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps) ** 2)

    return -np.sum(r) + penalty_weight * penalty


def _get_contact_edges(circles):
    """Get contact edges from Delaunay triangulation of centers."""
    centers = circles[:, :2]
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for k in range(3):
                a, b = simplex[k], simplex[(k + 1) % 3]
                edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        return None


def _initial_hex():
    """Hexagonal packing initial config."""
    cols = 5
    rows = 6
    positions = []
    r = 0.09
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 1:
                x = (col + 0.5) / cols + 0.5 / cols
            else:
                x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if x > 1 or y > 1:
                continue
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])


def _initial_grid():
    """Grid initial config."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    r = 0.5 / side * 0.9
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])


def _initial_corner_weighted(seed=0):
    """Random config with corner preference."""
    rng = np.random.default_rng(seed)
    positions = []
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
               [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.08])
    for _ in range(N - len(positions)):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        positions.append([x, y, rng.uniform(0.06, 0.10)])
    return np.array(positions[:N])


def _initial_voronoi(seed=0):
    """Initialize using Voronoi cells to estimate max possible radii."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))

    radii = np.zeros(N)
    for i in range(N):
        dists = np.linalg.norm(points - points[i], axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        wall_dist = np.min([points[i, 0], 1 - points[i, 0], points[i, 1], 1 - points[i, 1]])
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.95

    positions = np.column_stack([points[:, 0], points[:, 1], radii])
    return positions


def _optimize_single(xyz0, deadline, use_grad=True):
    """Optimize from a single initial config with adaptive penalty and optional analytic gradients."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])

    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))

    # Phase 1: Low penalty, focus on layout
    penalty = 5.0
    remaining = deadline - time.time()
    if remaining > 4:
        try:
            if use_grad:
                res = minimize(lambda p: _objective_grad(p, penalty, 1e-3)[0], params,
                              jac=lambda p: _objective_grad(p, penalty, 1e-3)[1],
                              method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            else:
                res = minimize(_objective_penalty, params, method='L-BFGS-B',
                              bounds=bounds, args=(penalty, 1e-3),
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective_penalty(params, penalty, 1e-3):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 2: Medium penalty
    remaining = deadline - time.time()
    if remaining > 4:
        penalty = 50.0
        try:
            if use_grad:
                res = minimize(lambda p: _objective_grad(p, penalty, 1e-4)[0], params,
                              jac=lambda p: _objective_grad(p, penalty, 1e-4)[1],
                              method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            else:
                res = minimize(_objective_penalty, params, method='L-BFGS-B',
                              bounds=bounds, args=(penalty, 1e-4),
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective_penalty(params, penalty, 1e-4):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 3: High penalty with Delaunay contact graph
    remaining = deadline - time.time()
    if remaining > 4:
        penalty = 500.0
        circles = np.column_stack([best_params[0::3], best_params[1::3], best_params[2::3]])
        contact_edges = _get_contact_edges(circles)
        try:
            res = minimize(_objective_delaunay, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4, contact_edges),
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective_delaunay(params, penalty, 1e-4, contact_edges):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass

    # Phase 4: Gradient ascent polish - directly maximize sum of radii
    remaining = deadline - time.time()
    if remaining > 5:
        try:
            current = best_params.copy()
            current_score = np.sum(current[2::3])
            best_polish = current.copy()
            best_polish_score = current_score

            # Use subgradient ascent on radii with constraint repair
            for _step in range(200):
                if time.time() > deadline - 2:
                    break

                # Compute constraint violations
                x = current[0::3]
                y = current[1::3]
                r = current[2::3]

                # Wall violations
                wall_viol = np.maximum(0, r - x) + np.maximum(0, x + r - 1) + \
                            np.maximum(0, r - y) + np.maximum(0, y + r - 1)

                # Pairwise overlaps
                idx_i, idx_j = np.triu_indices(N, k=1)
                dx = x[idx_i] - x[idx_j]
                dy = y[idx_i] - y[idx_j]
                dist = np.sqrt(dx * dx + dy * dy + 1e-12)
                overlap = np.maximum(0, r[idx_i] + r[idx_j] - dist)

                total_viol = np.sum(wall_viol) + np.sum(overlap)

                if total_viol < 1e-6:
                    # Try to increase radii
                    step_size = 0.001
                    current[2::3] += step_size
                    # Repair
                    circles = np.column_stack([current[0::3], current[1::3], current[2::3]])
                    circles = _fix_boundaries(circles)
                    current = circles.flatten()
                    score = np.sum(current[2::3])
                    if score > best_polish_score:
                        best_polish_score = score
                        best_polish = current.copy()
                else:
                    # Reduce radii where violations exist
                    current[2::3] -= 0.002 * wall_viol
                    for k in range(len(idx_i)):
                        if overlap[k] > 0:
                            current[3 * idx_i[k] + 2] -= 0.001 * overlap[k]
                            current[3 * idx_j[k] + 2] -= 0.001 * overlap[k]
                    current[2::3] = np.maximum(current[2::3], 0.01)
                    circles = np.column_stack([current[0::3], current[1::3], current[2::3]])
                    circles = _fix_boundaries(circles)
                    current = circles.flatten()
                    score = np.sum(current[2::3])
                    if score > best_polish_score:
                        best_polish_score = score
                        best_polish = current.copy()

            if best_polish_score > best_score:
                best_params = best_polish.copy()
                best_score = best_polish_score
        except Exception:
            pass

    # Final polish with full pairwise
    remaining = deadline - time.time()
    if remaining > 2:
        penalty = 1000.0
        try:
            res = minimize(_objective_penalty, best_params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5),
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective_penalty(best_params, penalty, 1e-5):
                best_params = res.x
                best_score = np.sum(best_params[2::3])
        except Exception:
            pass

    return best_params, best_score


def construct():
    """
    Pack 26 circles in the unit square, maximizing the sum of radii.
    Uses multi-start optimization with Delaunay-based contact graph and analytic gradients.
    """
    start_time = time.time()
    deadline = _start_time + _TIME_LIMIT

    # Generate multiple initial configurations
    inits = [
        _initial_hex(),
        _initial_grid(),
        _initial_corner_weighted(seed=42),
        _initial_corner_weighted(seed=123),
        _initial_corner_weighted(seed=999),
        _initial_voronoi(seed=7),
        _initial_voronoi(seed=1234),
    ]

    best_overall = None
    best_score_overall = -np.inf

    n_inits = len(inits)
    for i, xyz0 in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 5:
            break

        # Allocate time proportionally, giving more time to later starts
        alloc = min(remaining - 5, 20)
        local_deadline = time.time() + alloc

        # Use analytic gradient for some starts
        use_grad = (i % 2 == 0)

        params, score = _optimize_single(xyz0, local_deadline, use_grad=use_grad)

        # Validate
        circles = np.column_stack([params[0::3], params[1::3], params[2::3]])
        if _validate(circles):
            if score > best_score_overall:
                best_score_overall = score
                best_overall = circles.copy()
        else:
            # Try to fix
            circles_fixed = _fix_boundaries(circles)
            if _validate(circles_fixed):
                score_fixed = np.sum(circles_fixed[:, 2])
                if score_fixed > best_score_overall:
                    best_score_overall = score_fixed
                    best_overall = circles_fixed.copy()

    # Always ensure the result is valid by fixing boundaries and validating
    if best_overall is not None:
        repaired = _fix_boundaries(best_overall)
        if _validate(repaired):
            best_overall = repaired
        else:
            best_overall = None

    if best_overall is None:
        # Fallback: grid of 26 circles with small radii
        result = np.zeros((N, 3))
        side = 5
        for idx in range(N):
            i = idx % side
            j = idx // side
            result[idx, 0] = (i + 0.5) / side
            result[idx, 1] = (j + 0.5) / side
            result[idx, 2] = 0.5 / side * 0.8
        return result

    return best_overall

# EVOLVE-BLOCK-END
