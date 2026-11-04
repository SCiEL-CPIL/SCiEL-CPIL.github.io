"""Optimize 150 points on unit sphere to minimize logarithmic energy."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient (fast vectorized).
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient of energy w.r.t. x_i (in ambient R^3):
      dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # diff[i,j] = x_i - x_j
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero on diagonal

    iu = np.triu_indices(n, k=1)
    dist = np.sqrt(dist_sq[iu])
    dist = dist[dist > 1e-15]
    if len(dist) == 0:
        energy = 0.0
    else:
        energy = -math.fsum(np.log(dist).tolist())

    # Gradient: grad[i] = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_j (x_i - x_j) / dist_sq[i,j]  (diagonal excluded)
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-30
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_dist_sq, 0.0)

    # grad[i] = x_i * sum_j inv_dist_sq[i,j] - sum_j x_j * inv_dist_sq[i,j]
    row_sums = np.sum(inv_dist_sq, axis=1)
    weighted_sum = inv_dist_sq @ pts
    grad = pts * row_sums[:, np.newaxis] - weighted_sum

    return energy, grad


def _energy_only(pts):
    """Energy only, for fast inner loops."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    dist = np.sqrt(dist_sq[iu])
    dist = dist[dist > 1e-15]
    if len(dist) == 0:
        return 0.0
    return -math.fsum(np.log(dist).tolist())


def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _tangent_project(g, pts):
    """Project gradient to tangent space of sphere."""
    return g - np.sum(g * pts, axis=1, keepdims=True) * pts


def _fibonacci_sphere(n, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
        offset = rng.uniform(0, 1)
    else:
        offset = 0.0
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _random_sphere(n, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


def _lbfgs_optimize(pts, time_limit=5.0, ftol=1e-12, gtol=1e-10, maxiter=2000):
    """L-BFGS-B optimization with time limit."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())

    def objective_flat(x):
        if time.time() - start_time > time_limit:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e

    def grad_flat(x):
        if time.time() - start_time > time_limit:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = _tangent_project(g, p)
        return g_tangent.flatten()

    x0 = pts.flatten()

    result = minimize(
        objective_flat,
        x0,
        jac=grad_flat,
        method='L-BFGS-B',
        options={
            'maxiter': maxiter,
            'maxfun': 5000,
            'ftol': ftol,
            'gtol': gtol,
            'maxls': 50
        }
    )

    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def _polish(pts, time_limit=10.0):
    """Final high-precision polish with very tight tolerances."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())

    def objective_flat(x):
        if time.time() - start_time > time_limit:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e

    def grad_flat(x):
        if time.time() - start_time > time_limit:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = _tangent_project(g, p)
        return g_tangent.flatten()

    x0 = pts.flatten()

    result = minimize(
        objective_flat,
        x0,
        jac=grad_flat,
        method='L-BFGS-B',
        options={
            'maxiter': 10000,
            'maxfun': 20000,
            'ftol': 1e-15,
            'gtol': 1e-13,
            'maxls': 100
        }
    )

    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def _basin_hopping(pts, time_limit=30.0, rng=None):
    """Basin hopping: random perturbations + local minimization.
    
    This is a key strategy for escaping local minima in the logarithmic energy
    landscape. We perturb points randomly and re-optimize, keeping the best.
    """
    if rng is None:
        rng = np.random.RandomState()
    
    start_time = time.time()
    best_pts = _project_to_sphere(pts.copy())
    best_energy, _ = _energy_and_grad(best_pts)
    
    # Adaptive perturbation size
    base_perturb = 0.3
    min_perturb = 0.01
    perturb = base_perturb
    
    # Number of iterations we can afford
    # Each L-BFGS call takes ~1-3s, so we can do ~10-20 in 30s
    max_iter = 20
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Random perturbation in tangent space
        noise = rng.randn(N, 3) * perturb
        perturbed = _project_to_sphere(best_pts + noise)
        
        # Local optimization
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 5
        if remaining < 1.0:
            break
        
        candidate_pts, candidate_energy = _lbfgs_optimize(
            perturbed, time_limit=min(remaining, 3.0)
        )
        
        if candidate_energy < best_energy:
            best_energy = candidate_energy
            best_pts = candidate_pts
            # Reduce perturbation on success (annealing-like)
            perturb = max(min_perturb, perturb * 0.85)
        else:
            # Increase perturbation on failure to explore more
            perturb = min(base_perturb, perturb * 1.1)
    
    return best_pts, best_energy


def _adam_optimize(pts, time_limit=10.0):
    """Adam optimizer on the sphere for finer-tuning."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    lr = 0.005
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0

    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)

    while time.time() - start_time < time_limit:
        t += 1
        energy, grad = _energy_and_grad(pts)

        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        grad_tangent = _tangent_project(grad, pts)
        m_tangent = _tangent_project(m_hat, pts)

        pts = pts - lr * m_tangent / (np.sqrt(v_hat) + eps)
        pts = _project_to_sphere(pts)

        if t % 200 == 0 and np.linalg.norm(grad_tangent) < 1e-11:
            break

    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Phase 1: Multiple diverse starts with L-BFGS
    starts = []
    # Fibonacci with different offsets
    for i in range(4):
        starts.append(_fibonacci_sphere(N, seed=i))
    # Random starts
    for i in range(4):
        starts.append(_random_sphere(N, seed=200 + i))

    best_pts = None
    best_energy = float('inf')

    time_per_start = 4.0
    for idx, start_pts in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:  # Reserve 60s for basin hopping + polish
            break

        current_limit = min(time_per_start, deadline - elapsed - 60)
        if current_limit < 1.0:
            continue

        candidate_pts, candidate_energy = _lbfgs_optimize(
            start_pts, time_limit=current_limit
        )
        if candidate_energy < best_energy:
            best_energy = candidate_energy
            best_pts = candidate_pts

    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy, _ = _energy_and_grad(best_pts)

    # Phase 2: Basin hopping to escape local minima
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        bh_time = min(40.0, deadline - elapsed - 20)
        if bh_time > 5.0:
            rng = np.random.RandomState(123)
            candidate_pts, candidate_energy = _basin_hopping(
                best_pts, time_limit=bh_time, rng=rng
            )
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts

    # Phase 3: Adam fine-tuning
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        adam_time = min(8.0, deadline - elapsed - 10)
        if adam_time > 2.0:
            candidate_pts, candidate_energy = _adam_optimize(
                best_pts, time_limit=adam_time
            )
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts

    # Phase 4: Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 2:
        polish_time = min(12.0, deadline - elapsed - 1)
        if polish_time > 2.0:
            candidate_pts, candidate_energy = _polish(
                best_pts, time_limit=polish_time
            )
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts

    best_pts = _project_to_sphere(best_pts)

    # Validate: check for very close points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)

    if min_dist < 1e-9:
        return _fibonacci_sphere(N)

    # Final safety check
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)

    return best_pts
# EVOLVE-BLOCK-END
