import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _compute_score(points):
    """Compute the actual packing score (largest common radius)."""
    n = len(points)
    if n < 2:
        return 1.0 - np.linalg.norm(points[0]) if n == 1 else 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    return min(min_pair_dist, min_boundary)


def _hex_lattice_init(n, r_init=0.05):
    """Initialize points on a hexagonal lattice clipped to the unit circle."""
    spacing = 2.0 * r_init
    R_est = spacing * np.sqrt(n * np.sqrt(3.0) / (2.0 * np.pi))
    R_est *= 1.15
    k = int(np.ceil(R_est / spacing)) + 2
    points = []
    for j in range(-k, k + 1):
        y = j * spacing * np.sqrt(3.0) / 2.0
        if abs(y) > R_est:
            continue
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            if np.hypot(x, y) <= R_est:
                points.append((x, y))
    points = np.array(points)
    if len(points) == 0:
        return np.zeros((n, 2))
    if len(points) > n:
        radii = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(radii)
        points = points[indices[:n]]
    elif len(points) < n:
        while len(points) < n:
            angle = np.random.uniform(0, 2 * np.pi)
            rad = np.random.uniform(0, 0.95)
            points = np.vstack([points, [rad * np.cos(angle), rad * np.sin(angle)]])
    # Small perturbation to break symmetry
    points += np.random.normal(0, 1e-4, points.shape)
    return points


def _ring_init(n):
    """Initialize points in concentric rings with staggered phases."""
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    num_rings = int(1.0 / s) + 1
    points = [[0.0, 0.0]]
    for k in range(1, num_rings + 1):
        radius = k * s
        if radius > 0.98:
            break
        count = max(1, int(2 * np.pi * radius / s))
        phase = (k % 2) * np.pi / count
        for m in range(count):
            if len(points) >= n:
                break
            angle = 2 * np.pi * m / count + phase
            points.append([radius * np.cos(angle), radius * np.sin(angle)])
    points = np.array(points)
    if len(points) < n:
        while len(points) < n:
            x, y = np.random.uniform(-0.95, 0.95, 2)
            if x * x + y * y <= 0.9:
                points = np.vstack([points, [[x, y]]])
    elif len(points) > n:
        points = points[:n]
    points += np.random.normal(0, 1e-4, points.shape)
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s limit

    # Precompute upper-triangle pair indices (used throughout)
    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    n_pairs = len(i_idx)

    # Generate candidate initializations
    inits = [_hex_lattice_init(N, 0.05), _ring_init(N)]
    # Add a second hex init with slightly different spacing for diversity
    inits.append(_hex_lattice_init(N, 0.048))

    best_points = None
    best_score = -np.inf
    for init_points in inits:
        score = _compute_score(init_points)
        if score > best_score:
            best_points = init_points.copy()
            best_score = score

    if best_points is None:
        best_points = _hex_lattice_init(N, 0.05)
        best_score = _compute_score(best_points)

    def make_objective_and_grad(beta):
        def objective_and_grad(x):
            p = x.reshape(-1, 2)
            # Pair distances (only upper triangle)
            vecs = p[i_idx] - p[j_idx]
            pair_dists = np.sqrt(np.sum(vecs**2, axis=1)) / 2.0
            # Boundary distances
            norms = np.linalg.norm(p, axis=1)
            bound_dists = 1.0 - norms

            # Log-sum-exp for numerical stability
            z_pairs = -beta * pair_dists
            z_bounds = -beta * bound_dists
            max_z = max(np.max(z_pairs), np.max(z_bounds))

            exp_pairs = np.exp(z_pairs - max_z)
            exp_bounds = np.exp(z_bounds - max_z)
            sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
            log_sum_exp = max_z + np.log(sum_exp)

            val = log_sum_exp / beta

            # Gradient
            w_pairs = exp_pairs / sum_exp
            w_bounds = exp_bounds / sum_exp

            grad = np.zeros_like(x)

            # Boundary gradient: d(1 - ||p_i||)/dp_i = -p_i / ||p_i||
            norms_safe = norms.copy()
            norms_safe[norms_safe == 0] = 1.0
            grad += -p / norms_safe[:, np.newaxis] * w_bounds[:, np.newaxis]

            # Pair distance gradient
            # d(||p_i - p_j||/2)/dp_i = (p_i - p_j) / (2*||p_i - p_j||)
            pair_full_dists = np.sqrt(np.sum(vecs**2, axis=1))
            pair_full_dists_safe = np.where(pair_full_dists > 1e-12, pair_full_dists, 1.0)
            coeff = w_pairs / pair_full_dists_safe

            # Accumulate into grad[i] += coeff * (p_i - p_j), grad[j] -= coeff * (p_i - p_j)
            np.add.at(grad, i_idx, coeff[:, np.newaxis] * vecs)
            np.add.at(grad, j_idx, -coeff[:, np.newaxis] * vecs)

            return val, grad
        return objective_and_grad

    bounds = [(-1.05, 1.05)] * (2 * N)

    # Annealing schedule: increasing beta, decreasing iterations
    beta_schedule = [
        (15.0, 120),
        (40.0, 100),
        (80.0, 80),
        (160.0, 60),
        (320.0, 50),
        (640.0, 40),
    ]

    for beta, max_iter in beta_schedule:
        if time.time() > deadline - 25:
            break

        obj_grad = make_objective_and_grad(beta)

        try:
            res = minimize(obj_grad, best_points.flatten(), jac=True,
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': 1e-14,
                                    'gtol': 1e-10, 'disp': False})
            if res.x is not None:
                candidate = res.x.reshape(-1, 2)
                radii = np.hypot(candidate[:, 0], candidate[:, 1])
                if np.all(radii < 1.0) and np.all(np.isfinite(candidate)):
                    score = _compute_score(candidate)
                    if score > best_score:
                        best_points = candidate
                        best_score = score
        except Exception:
            pass

    # Local refinement: nudge points that are close to their nearest neighbor
    # or close to the boundary, to break shallow local minima.
    if time.time() < deadline - 30:
        for _ in range(3):
            if time.time() > deadline - 30:
                break
            p = best_points.copy()
            # Compute all pairwise distances
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            # Find nearest neighbor for each point
            nn_dist = np.min(dists, axis=1)
            nn_idx = np.argmin(dists, axis=1)
            # Points with very close neighbors get nudged
            close_mask = nn_dist < 2.0 * best_score * 1.05
            if np.any(close_mask):
                for idx in np.where(close_mask)[0]:
                    other = nn_idx[idx]
                    d = p[idx] - p[other]
                    dn = np.linalg.norm(d)
                    if dn > 1e-10:
                        p[idx] += d / dn * best_score * 0.05
                # Also nudge boundary-close points inward
                norms = np.linalg.norm(p, axis=1)
                bound_close = norms > 1.0 - best_score * 1.05
                for idx in np.where(bound_close)[0]:
                    p[idx] *= (1.0 - best_score * 0.05)
            # Boundary-only nudge
            norms = np.linalg.norm(p, axis=1)
            bound_close = norms > 1.0 - best_score * 1.05
            if np.any(bound_close):
                for idx in np.where(bound_close)[0]:
                    p[idx] *= (1.0 - best_score * 0.05)

            # Re-optimize from nudged start
            obj_grad = make_objective_and_grad(640.0)
            try:
                res = minimize(obj_grad, p.flatten(), jac=True,
                               method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 60, 'ftol': 1e-14,
                                        'gtol': 1e-10, 'disp': False})
                if res.x is not None:
                    candidate = res.x.reshape(-1, 2)
                    radii = np.hypot(candidate[:, 0], candidate[:, 1])
                    if np.all(radii < 1.0) and np.all(np.isfinite(candidate)):
                        score = _compute_score(candidate)
                        if score > best_score:
                            best_points = candidate
                            best_score = score
            except Exception:
                pass

    # Final high-precision polish
    if time.time() < deadline - 8:
        obj_grad = make_objective_and_grad(2000.0)
        try:
            res = minimize(obj_grad, best_points.flatten(), jac=True,
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 40, 'ftol': 1e-16,
                                    'gtol': 1e-12, 'disp': False})
            if res.x is not None:
                candidate = res.x.reshape(-1, 2)
                radii = np.hypot(candidate[:, 0], candidate[:, 1])
                if np.all(radii < 1.0) and np.all(np.isfinite(candidate)):
                    score = _compute_score(candidate)
                    if score > best_score:
                        best_points = candidate
                        best_score = score
        except Exception:
            pass

    # Final validity check
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(radii >= 1.0):
        max_r = np.max(radii)
        if max_r > 0:
            scale = 0.9999 / max_r
            best_points *= scale

    return best_points
