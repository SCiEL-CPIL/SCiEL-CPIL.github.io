import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 138.0  # Safety margin before 150s kill


def _normalize(points):
    pts = np.asarray(points, dtype=float)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _facet_alphas(points):
    """Return per-facet angular circumradius (radians). Returns empty array if hull fails."""
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.array([])
    if simplices.size == 0:
        return np.array([])
    tri = points[simplices]
    p0 = tri[:, 0, :]
    p1 = tri[:, 1, :]
    p2 = tri[:, 2, :]
    nrm = np.cross(p1 - p0, p2 - p0)
    nn = np.linalg.norm(nrm, axis=1)
    mask = nn > 1e-12
    if not np.any(mask):
        return np.array([])
    d = np.abs(nrm[mask] @ p0[mask].T) / nn[mask]
    d = np.clip(d, 0.0, 1.0)
    return np.arccos(d)


def _covering_radius(points):
    """Exact covering radius in radians (max facet circumradius)."""
    alphas = _facet_alphas(points)
    if alphas.size == 0:
        return np.pi
    return float(np.max(alphas))


def _min_pairwise_dist(points):
    """Minimum squared Euclidean distance between any pair of points."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    return float(np.min(dist_sq))


def _soft_max(values, beta):
    """Log-sum-exp based soft max for smooth optimization."""
    if values.size == 0:
        return np.pi
    m = np.max(values)
    return m + np.log(np.sum(np.exp(values - m))) / beta


def _density_aware_relaxation(points, steps=1000, alpha=0.08, time_limit=None,
                               start_time=None, force_exponent=2.0):
    """
    Relax points using repulsive forces scaled by local covering density.
    Adaptive alpha: larger early, smaller late for convergence.
    """
    if start_time is None:
        start_time = time.time()
    points = points.copy()
    n = points.shape[0]

    for step in range(steps):
        if time_limit is not None and time.time() - start_time > time_limit:
            break

        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        mask = dist_sq > 1e-12

        dist_sq_safe = dist_sq.copy()
        np.fill_diagonal(dist_sq_safe, np.inf)
        nn_dist_sq = np.min(dist_sq_safe, axis=1)
        nn_dist = np.sqrt(np.maximum(nn_dist_sq, 1e-12))

        mean_nn = np.mean(nn_dist)
        if mean_nn < 1e-12:
            mean_nn = 1e-12
        density_scale = (mean_nn / nn_dist) ** 0.5
        density_scale = np.clip(density_scale, 0.5, 3.0)

        inv_dist = np.zeros_like(dist_sq)
        inv_dist[mask] = 1.0 / (dist_sq[mask] ** (force_exponent / 2.0))

        forces = np.sum(diff * inv_dist[:, :, np.newaxis], axis=1)
        forces *= density_scale[:, np.newaxis]

        # Adaptive step size: cosine decay for smoother convergence
        progress = step / max(steps - 1, 1)
        step_size = alpha * (0.3 + 0.7 * np.cos(np.pi * progress / 2.0))
        points += step_size * forces
        points = _normalize(points)

    return points


def _spherical_to_cartesian(params):
    theta = params[0::2]
    phi = params[1::2]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])


def _cartesian_to_spherical(pts):
    theta = np.arccos(np.clip(pts[:, 2], -1.0, 1.0))
    phi = np.atan2(pts[:, 1], pts[:, 0])
    params = np.zeros(2 * N)
    params[0::2] = theta
    params[1::2] = phi
    return params


def _smooth_objective(params, beta=50.0):
    """Smooth objective using log-sum-exp over facet circumradii."""
    if time.time() > _DEADLINE:
        return 1e6
    pts = _spherical_to_cartesian(params)
    if not np.all(np.isfinite(pts)):
        return 1e6
    alphas = _facet_alphas(pts)
    if alphas.size == 0:
        return 1e6
    return _soft_max(alphas, beta)


def _tammes_objective(params):
    """Maximize minimum pairwise distance (minimize negative of it)."""
    if time.time() > _DEADLINE:
        return 1e6
    pts = _spherical_to_cartesian(params)
    if not np.all(np.isfinite(pts)):
        return 1e6
    return -_min_pairwise_dist(pts)


def _fibonacci_spiral():
    """Generate initial Fibonacci spiral configuration."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def construct():
    start_time = time.time()

    def time_remaining():
        return _DEADLINE - time.time()

    # 1. Initial Configuration: Fibonacci Spiral
    points = _fibonacci_spiral()
    best_points = points.copy()
    best_radius = _covering_radius(best_points)

    # 2. Multi-exponent Density-Aware Repulsive Relaxation
    for force_exp in [2.0, 2.5, 3.0]:
        if time_remaining() < 15:
            break
        points = _density_aware_relaxation(points, steps=2000, alpha=0.08,
                                           time_limit=min(8, time_remaining() * 0.3),
                                           start_time=start_time,
                                           force_exponent=force_exp)
        current_radius = _covering_radius(points)
        if current_radius < best_radius:
            best_points = points.copy()
            best_radius = current_radius

    # 3. Tammes-style optimization: maximize min pairwise distance
    params0 = _cartesian_to_spherical(best_points)
    if time_remaining() > 25:
        try:
            result = minimize(
                _tammes_objective,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            new_pts = _spherical_to_cartesian(result.x)
            new_rad = _covering_radius(new_pts)
            if new_rad < best_radius:
                best_points = new_pts
                best_radius = new_rad
                params0 = result.x
        except Exception:
            pass

    # 4. Coarse-to-fine Log-Sum-Exp optimization
    for beta in [30.0, 80.0, 200.0, 500.0, 1000.0]:
        if time_remaining() < 8:
            break
        try:
            result = minimize(
                lambda p: _smooth_objective(p, beta=beta),
                params0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            new_pts = _spherical_to_cartesian(result.x)
            new_rad = _covering_radius(new_pts)
            if new_rad < best_radius:
                best_points = new_pts
                best_radius = new_rad
                params0 = result.x
        except Exception:
            pass

    # 5. Nelder-Mead Polish with Exact Objective
    if time_remaining() > 12:
        try:
            result = minimize(
                lambda p: _covering_radius(_spherical_to_cartesian(p)),
                params0,
                method='Nelder-Mead',
                options={'maxiter': 4000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_radius:
                best_points = _spherical_to_cartesian(result.x)
                best_radius = result.fun
                params0 = _cartesian_to_spherical(best_points)
        except Exception:
            pass

    # 6. Random Restarts with Multi-Stage Optimization
    restart_count = 0
    noise_scales = [0.08, 0.12, 0.18, 0.25, 0.4]
    while time_remaining() > 18:
        restart_count += 1
        noise_scale = noise_scales[restart_count % len(noise_scales)]

        noise = np.random.normal(0, noise_scale, size=best_points.shape)
        perturbed = best_points + noise
        perturbed = _normalize(perturbed)

        force_exp = [2.0, 2.5, 3.0][restart_count % 3]
        relaxed = _density_aware_relaxation(perturbed, steps=400, alpha=0.06,
                                            time_limit=min(4, time_remaining() * 0.2),
                                            start_time=start_time,
                                            force_exponent=force_exp)
        rad = _covering_radius(relaxed)

        if rad < best_radius:
            best_points = relaxed
            best_radius = rad
            params0 = _cartesian_to_spherical(best_points)

            # Tammes optimization from new start
            if time_remaining() > 12:
                try:
                    result = minimize(
                        _tammes_objective,
                        params0,
                        method='L-BFGS-B',
                        options={'maxiter': 250, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                    )
                    new_pts = _spherical_to_cartesian(result.x)
                    new_rad = _covering_radius(new_pts)
                    if new_rad < best_radius:
                        best_points = new_pts
                        best_radius = new_rad
                        params0 = result.x
                except Exception:
                    pass

            # Smooth surrogate optimization (coarse-to-fine)
            for beta in [80.0, 300.0, 800.0]:
                if time_remaining() < 6:
                    break
                try:
                    result = minimize(
                        lambda p: _smooth_objective(p, beta=beta),
                        params0,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                    )
                    new_pts = _spherical_to_cartesian(result.x)
                    new_rad = _covering_radius(new_pts)
                    if new_rad < best_radius:
                        best_points = new_pts
                        best_radius = new_rad
                        params0 = result.x
                except Exception:
                    pass

            # Nelder-Mead polish
            if time_remaining() > 8:
                try:
                    result = minimize(
                        lambda p: _covering_radius(_spherical_to_cartesian(p)),
                        params0,
                        method='Nelder-Mead',
                        options={'maxiter': 2500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    if result.fun < best_radius:
                        best_points = _spherical_to_cartesian(result.x)
                        best_radius = result.fun
                        params0 = _cartesian_to_spherical(best_points)
                except Exception:
                    pass

    # 7. Final High-Precision Polish
    if time_remaining() > 8:
        params0 = _cartesian_to_spherical(best_points)
        try:
            result = minimize(
                lambda p: _smooth_objective(p, beta=2000.0),
                params0,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            new_pts = _spherical_to_cartesian(result.x)
            new_rad = _covering_radius(new_pts)
            if new_rad < best_radius:
                best_points = new_pts
                best_radius = new_rad
        except Exception:
            pass

    if time_remaining() > 4:
        params0 = _cartesian_to_spherical(best_points)
        try:
            result = minimize(
                lambda p: _covering_radius(_spherical_to_cartesian(p)),
                params0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_radius:
                best_points = _spherical_to_cartesian(result.x)
                best_radius = result.fun
        except Exception:
            pass

    return best_points
