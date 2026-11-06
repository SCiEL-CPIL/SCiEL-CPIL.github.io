"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 150
TIME_LIMIT = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _log_energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    return -math.fsum(log_dists.tolist())


def _fibonacci_sphere(n, seed=0.0):
    """Generate n points on the sphere using a Fibonacci spiral with an offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng=None):
    """Generate n random points on the unit sphere."""
    if rng is None:
        rng = np.random.default_rng()
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _energy_and_grad_flat(flat):
    """
    Objective and gradient for L-BFGS-B on flat 3N vector.
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient w.r.t. x_i: sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    pts = flat.reshape(N, 3)
    n = N
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dists_sq = np.sum(diff * diff, axis=-1)  # (n, n)
    np.fill_diagonal(dists_sq, 1.0)
    
    # Energy: -sum_{i<j} log(dist_ij)
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(np.sqrt(dists_sq[iu]))
    energy = -math.fsum(log_dists.tolist())
    
    # Gradient: for each i, grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j!=i} diff[i,j] / dists_sq[i,j]
    # dists_sq has diagonal 1.0, so we can compute sum over all j and subtract i term (which is 0)
    inv_dists_sq = 1.0 / dists_sq  # (n, n)
    # grad = sum_j diff[i,j] * inv_dists_sq[i,j] for each i
    # diff is (n, n, 3), inv_dists_sq is (n, n)
    grad = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)  # (n, 3)
    grad_flat = grad.flatten()
    
    return energy, grad_flat


def _refine_lbfgs(points, maxiter=200):
    """
    Refine points using L-BFGS-B on the flat 3N representation.
    The optimizer may move points slightly off the sphere; we normalize at the end.
    """
    flat = points.flatten()
    result = minimize(
        _energy_and_grad_flat,
        flat,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-10}
    )
    refined = result.x.reshape(N, 3)
    return _normalize(refined)


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_energy = float('inf')
    best_points = _fibonacci_sphere(N)

    # 1. Evaluate initial Fibonacci
    try:
        e = _log_energy(best_points)
        if e < best_energy:
            best_energy = e
    except Exception:
        pass

    # 2. Refine Fibonacci start with L-BFGS
    rng = np.random.default_rng(42)
    try:
        refined = _refine_lbfgs(best_points, maxiter=300)
        e = _log_energy(refined)
        if e < best_energy:
            best_energy = e
            best_points = refined
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 30:
        return _normalize(best_points)

    # 3. Basin hopping: perturb current best, refine with L-BFGS
    #    Multiple restarts from random and perturbed configurations
    num_restarts = 200

    for k in range(num_restarts):
        if time.time() > deadline - 15:
            break

        # Choose starting point: mix of random and perturbed best
        if k % 3 == 0:
            # Random start
            pts = _random_sphere(N, rng)
        elif k % 3 == 1:
            # Perturbed best (large perturbation)
            pts = best_points + rng.standard_normal((N, 3)) * 0.5
            pts = _normalize(pts)
        else:
            # Perturbed best (smaller perturbation)
            pts = best_points + rng.standard_normal((N, 3)) * 0.2
            pts = _normalize(pts)

        try:
            refined_pts = _refine_lbfgs(pts, maxiter=150)
            e = _log_energy(refined_pts)

            if e < best_energy:
                best_energy = e
                best_points = refined_pts

                # If it's good, refine more
                if time.time() < deadline - 20:
                    further_refined = _refine_lbfgs(refined_pts, maxiter=200)
                    e2 = _log_energy(further_refined)
                    if e2 < best_energy:
                        best_energy = e2
                        best_points = further_refined
        except Exception:
            continue

    # 4. Final polish if time permits
    if time.time() < deadline - 5:
        try:
            final_pts = _refine_lbfgs(best_points, maxiter=500)
            e = _log_energy(final_pts)
            if e < best_energy:
                best_energy = e
                best_points = final_pts
        except Exception:
            pass

    return _normalize(best_points)

# EVOLVE-BLOCK-END
