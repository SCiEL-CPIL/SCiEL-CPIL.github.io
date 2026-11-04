"""Optimize 150 points on unit sphere to minimize logarithmic energy."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(points):
    """Compute log(1/d) energy using math.fsum for accuracy."""
    n = len(points)
    if n < 2:
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    return -math.fsum(log_dists.tolist())


def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms

    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)

    # E = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    inv_dists_sq = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    inv_dists_sq[mask] = 1.0 / dists_sq[mask]

    sum_weights = np.sum(inv_dists_sq, axis=1)
    weighted_sum = inv_dists_sq @ points

    grads = weighted_sum - points * sum_weights[:, np.newaxis]

    # Project gradient to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points

    # Compute energy with math.fsum
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    energy = -math.fsum(log_dists.tolist())

    return energy, grads_tangent.flatten()


def _energy_only(x):
    """Energy only for line search."""
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms
    return _log_energy(points)


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=42):
    """Generate random uniform points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _tangent_vector(point, rng):
    """Generate a random unit tangent vector at point on sphere."""
    rand_vec = rng.randn(3)
    tangent = rand_vec - np.dot(rand_vec, point) * point
    norm = np.linalg.norm(tangent)
    if norm < 1e-15:
        rand_vec = rng.randn(3)
        tangent = rand_vec - np.dot(rand_vec, point) * point
        norm = np.linalg.norm(tangent)
    return tangent / norm


def _lbfgs_optimize(points, max_iter=3000, ftol=1e-12, gtol=1e-10, time_limit=20.0):
    """L-BFGS-B optimization with time limit."""
    start = time.time()
    x0 = points.flatten()
    e0 = _log_energy(points)
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'maxfun': max_iter * 10,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 50
            }
        )
        if res.success or res.fun < e0:
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _log_energy(p_res)
            if e_check < e0 and time.time() - start < time_limit:
                return p_res
    except Exception:
        pass
    return points


def _adam_optimize(points, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
                   max_iter=5000, time_limit=15.0):
    """Adam optimizer on the sphere with tangent projection."""
    start = time.time()
    pts = points.copy()
    best_e = _log_energy(pts)
    best_pts = pts.copy()

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)

    for t in range(1, max_iter + 1):
        if time.time() - start > time_limit:
            break

        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)

        weights = np.zeros_like(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        weights[mask] = 1.0 / dists_sq[mask]

        sum_weights = np.sum(weights, axis=1)
        weighted_sum = weights @ pts
        grads = weighted_sum - pts * sum_weights[:, np.newaxis]

        # Project to tangent
        dot_products = np.sum(grads * pts, axis=1, keepdims=True)
        grads_tangent = grads - dot_products * pts

        # Adam update
        m = beta1 * m + (1 - beta1) * grads_tangent
        v = beta2 * v + (1 - beta2) * grads_tangent ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        update = m_hat / (np.sqrt(v_hat) + eps)
        new_pts = pts - lr * update
        new_pts = _project_to_sphere(new_pts)

        e_new = _log_energy(new_pts)
        if e_new < best_e:
            best_e = e_new
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            lr *= 0.95

    return best_pts


def _gradient_descent_sphere(points, lr=0.01, max_iter=10000, time_limit=15.0):
    """Gradient descent on the sphere with adaptive step size."""
    start = time.time()
    pts = points.copy()
    best_e = _log_energy(pts)
    best_pts = pts.copy()

    for it in range(max_iter):
        if time.time() - start > time_limit:
            break

        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)

        weights = np.zeros_like(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        weights[mask] = 1.0 / dists_sq[mask]

        sum_weights = np.sum(weights, axis=1)
        weighted_sum = weights @ pts
        grads = weighted_sum - pts * sum_weights[:, np.newaxis]

        dot_products = np.sum(grads * pts, axis=1, keepdims=True)
        grads_tangent = grads - dot_products * pts

        grad_norm = np.linalg.norm(grads_tangent)
        if grad_norm < 1e-12:
            break

        step = lr / (1.0 + grad_norm * 0.1)
        new_pts = pts - step * grads_tangent
        new_pts = _project_to_sphere(new_pts)

        e_new = _log_energy(new_pts)
        if e_new < best_e:
            best_e = e_new
            best_pts = new_pts.copy()
            pts = new_pts
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.9

    return best_pts


def _basin_hopping(init_points, deadline, rng, max_iter=30, T_init=0.15, T_min=1e-8, cooling=0.95):
    """
    Basin hopping:
    1. Local minimize
    2. Perturb a random subset of points
    3. Local minimize
    4. Accept if better (Metropolis)
    """
    current_points, current_energy = _local_minimize(init_points, deadline, maxiter=200)
    best_points = current_points.copy()
    best_energy = current_energy

    T = T_init

    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        if T < T_min:
            break

        # Perturb a random subset of points
        n_perturb = rng.randint(2, 10)
        idxs = rng.choice(N, n_perturb, replace=False)

        new_points = current_points.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_points[idx], rng)
            step_size = T
            new_point = new_points[idx] + step_size * tangent
            new_point = new_point / np.linalg.norm(new_point)
            new_points[idx] = new_point

        # Local minimize the perturbed configuration
        new_points, new_energy = _local_minimize(new_points, deadline, maxiter=100)

        # Metropolis criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()

        T *= cooling

    return best_points, best_energy


def _local_minimize(points, deadline, maxiter=500, ftol=1e-12, gtol=1e-10):
    """Run L-BFGS-B to find local minimum."""
    x0 = points.flatten()

    # Check if we have time
    if time.time() > deadline - 1.0:
        return points, _log_energy(points)

    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 20, 'ftol': ftol, 'gtol': gtol, 'maxls': 40}
        )
        # Check time again
        if time.time() > deadline - 1.0:
            return points, _log_energy(points)

        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _log_energy(p_res)
        return p_res, e_check
    except Exception:
        return points, _log_energy(points)


def _polish_high_precision(pts, max_time):
    """High-precision final polish with L-BFGS at tight tolerances."""
    pts = _project_to_sphere(np.array(pts, dtype=np.float64))
    t0 = time.time()

    def obj(x):
        if time.time() - t0 > max_time:
            return np.inf
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e

    def grad(x):
        if time.time() - t0 > max_time:
            return np.zeros_like(x)
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        _, g = _energy_and_grad(p)
        return g

    x0 = pts.flatten()
    try:
        res = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 50000}
        )
        pts_opt = _project_to_sphere(res.x.reshape(N, 3))
        e_opt = _log_energy(pts_opt)
        return pts_opt, e_opt
    except Exception:
        return pts, _log_energy(pts)


def _multi_start_optimize(points, time_limit=25.0):
    """Try multiple perturbation strategies and pick the best."""
    start = time.time()
    best_pts = points.copy()
    best_e = _log_energy(best_pts)

    for scale in [0.05, 0.1, 0.2]:
        if time.time() - start > time_limit - 10.0:
            break

        rng = np.random.RandomState(int(time.time() * 1000) % 100000)
        perturbed = points + rng.randn(N, 3) * scale
        perturbed = _project_to_sphere(perturbed)

        gd_result = _gradient_descent_sphere(perturbed, lr=0.01, max_iter=2000,
                                              time_limit=time_limit - (time.time() - start) - 5.0)
        e_gd = _log_energy(gd_result)

        if e_gd < best_e:
            best_e = e_gd
            best_pts = gd_result.copy()

    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Multiple initializations
    inits = [
        _fibonacci_sphere(N),
        _random_uniform_sphere(N, seed=1),
        _random_uniform_sphere(N, seed=2),
        _random_uniform_sphere(N, seed=7),
        _random_uniform_sphere(N, seed=42),
        _random_uniform_sphere(N, seed=99),
    ]

    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)

    # First pass: L-BFGS-B on all initializations
    for idx, init in enumerate(inits):
        if time.time() > deadline - 50.0:
            break

        result = _lbfgs_optimize(init, max_iter=2000, ftol=1e-12, gtol=1e-10,
                                  time_limit=min(10.0, deadline - time.time() - 40.0))
        e_check = _log_energy(result)
        if e_check < best_energy:
            best_energy = e_check
            best_points = result.copy()

    # Second pass: Basin hopping for escaping local minima
    if time.time() < deadline - 35.0:
        rng = np.random.RandomState(12345)
        bh_result, bh_energy = _basin_hopping(best_points, deadline - time.time() - 30.0,
                                              rng, max_iter=30, T_init=0.15)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_points = bh_result.copy()

    # Third pass: Adam fine-tuning
    if time.time() < deadline - 20.0:
        adam_result = _adam_optimize(best_points, lr=0.001, max_iter=5000,
                                      time_limit=deadline - time.time() - 15.0)
        e_adam = _log_energy(adam_result)
        if e_adam < best_energy:
            best_energy = e_adam
            best_points = adam_result.copy()

    # Fourth pass: aggressive L-BFGS-B polishing
    if time.time() < deadline - 12.0:
        polished = _lbfgs_optimize(best_points, max_iter=5000, ftol=1e-14, gtol=1e-12,
                                    time_limit=deadline - time.time() - 8.0)
        e_polish = _log_energy(polished)
        if e_polish < best_energy:
            best_energy = e_polish
            best_points = polished.copy()

    # Fifth pass: multi-start if time remains
    if time.time() < deadline - 8.0:
        multi_result = _multi_start_optimize(best_points,
                                              time_limit=deadline - time.time() - 5.0)
        e_multi = _log_energy(multi_result)
        if e_multi < best_energy:
            best_energy = e_multi
            best_points = multi_result.copy()

    # Final ultra-tight polish
    if time.time() < deadline - 4.0:
        final_polished, e_final = _polish_high_precision(best_points, deadline - time.time() - 2.0)
        if e_final < best_energy:
            best_points = final_polished.copy()

    best_points = _project_to_sphere(best_points)

    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)

    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    if np.min(dists_sq[iu]) < 1e-18:
        return _fibonacci_sphere(N)

    return best_points
# EVOLVE-BLOCK-END
