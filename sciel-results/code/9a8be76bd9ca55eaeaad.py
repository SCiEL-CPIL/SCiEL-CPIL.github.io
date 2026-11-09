import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _raw_log_energy(pts):
    """Compute the exact logarithmic energy using math.fsum for accuracy."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(N, k=1)
    return -math.fsum(np.log(dist[iu]).tolist())


def _energy_and_grad_sphere(x_flat):
    """
    Energy and tangent-space gradient on the sphere.
    We project x onto the sphere, compute the Euclidean gradient, then
    project the gradient to the tangent space.
    """
    x = x_flat.reshape(N, 3)
    x = _project_to_sphere(x)

    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)

    mask = ~np.eye(N, dtype=bool)
    log_dists = np.log(dist)
    log_dists[~mask] = 0.0
    E_log = -0.5 * np.sum(log_dists)

    # Euclidean gradient of E_log
    # dE/dx_i = -0.5 * sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    grad_log = -0.5 * np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)

    # Project gradient to tangent space: g_t = g - (g . x) x
    g_tangent = grad_log - np.sum(grad_log * x, axis=1, keepdims=True) * x

    return E_log, g_tangent.flatten()


def _project_to_sphere(pts):
    """Project points exactly onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_points(n=N, offset=0.0):
    """Generate Fibonacci spiral points on the unit sphere with optional offset."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)


def _random_sphere_points(n=N, seed=42):
    """Generate random points on the unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _optimize_lgbfgs_sphere(x0, t_start, time_limit, maxiter=300, ftol=1e-10, gtol=1e-6):
    """
    Run L-BFGS-B on the sphere using tangent-space gradients.
    """
    elapsed = time.time() - t_start
    if elapsed > time_limit - 5:
        return None, None

    try:
        res = minimize(
            _energy_and_grad_sphere,
            x0.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': maxiter,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 50
            }
        )

        candidate = _project_to_sphere(res.x.reshape(N, 3))
        cand_energy = _raw_log_energy(candidate)
        return candidate, cand_energy
    except Exception:
        return None, None


def _simulated_annealing_kick(pts, t_start, time_limit, max_iter=300, t_init=0.005, t_min=1e-6, rng=None):
    """
    A short simulated-annealing phase to kick the configuration out of shallow local minima.
    """
    if rng is None:
        rng = np.random.default_rng()

    pts = _project_to_sphere(pts.copy())
    current_energy = _raw_log_energy(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    T = t_init
    cooling = (t_min / t_init) ** (1.0 / max_iter)

    for it in range(max_iter):
        if time.time() > time_limit - 5:
            break

        # Small random perturbation in tangent directions
        noise_scale = 0.005 * (1.0 + 0.5 * it / max_iter)
        noise = rng.standard_normal((N, 3))
        # Project noise to tangent space
        noise_tangent = noise - np.sum(noise * pts, axis=1, keepdims=True) * pts
        new_pts = _project_to_sphere(pts + noise_scale * noise_tangent)

        new_energy = _raw_log_energy(new_pts)
        delta = new_energy - current_energy

        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()

        T *= cooling

    return best_pts, best_energy


def _polish_sphere(x, t_start, time_limit, max_iter=500, ftol=1e-14, gtol=1e-10):
    """Polish a configuration with tighter tolerances on the sphere."""
    elapsed = time.time() - t_start
    if elapsed > time_limit - 5:
        return x

    try:
        res = minimize(
            _energy_and_grad_sphere,
            x.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': maxiter,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 100
            }
        )

        candidate = _project_to_sphere(res.x.reshape(N, 3))
        return candidate
    except Exception:
        return x


def construct():
    t_start = time.time()
    time_limit = 115.0  # Safety margin before 120s kill

    # Generate diverse starting points
    starts = []

    # Fibonacci points with different offsets
    for offset in [0.0, 0.1, 0.2, 0.3, 0.5]:
        starts.append(_fibonacci_points(N, offset=offset))

    # Random points with different seeds
    for seed in [42, 123, 456, 789, 101112]:
        starts.append(_random_sphere_points(N, seed=seed))

    best_x = starts[0].copy()
    best_energy = _raw_log_energy(best_x)

    # Try each starting point with L-BFGS-B on the sphere
    for i, x0 in enumerate(starts):
        elapsed = time.time() - t_start
        if elapsed > time_limit - 30:
            break

        candidate, cand_energy = _optimize_lgbfgs_sphere(x0, t_start, time_limit, maxiter=300, ftol=1e-10, gtol=1e-6)

        if candidate is not None and cand_energy < best_energy:
            best_x = candidate
            best_energy = cand_energy

    # Optional: short SA kick from the best configuration if time remains
    elapsed = time.time() - t_start
    if elapsed < time_limit - 30:
        sa_x, sa_energy = _simulated_annealing_kick(
            best_x,
            t_start,
            time_limit - 5,
            max_iter=200,
            t_init=0.005,
            t_min=1e-6
        )
        if sa_energy < best_energy:
            best_x = sa_x
            best_energy = sa_energy

    # Polish the best result
    elapsed = time.time() - t_start
    if elapsed < time_limit - 20:
        best_x = _polish_sphere(best_x, t_start, time_limit, max_iter=500, ftol=1e-14, gtol=1e-10)
        best_energy = _raw_log_energy(best_x)

    # Final high-precision polish if time remains
    elapsed = time.time() - t_start
    if elapsed < time_limit - 10:
        best_x = _polish_sphere(best_x, t_start, time_limit, max_iter=300, ftol=1e-16, gtol=1e-12)

    # Final normalization
    best_x = _project_to_sphere(best_x)

    return best_x
