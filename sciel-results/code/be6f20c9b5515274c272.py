"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    return -np.sum(log_dists)


def _energy_and_grad(x_flat):
    """Energy and gradient for L-BFGS-B. x_flat is shape (N*3,)."""
    n = N
    points = x_flat.reshape(n, 3)
    
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    # Energy = -sum_{i<j} log(d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # Using dists_sq to avoid sqrt: log(d) = 0.5 * log(d^2)
    log_dists_sq = np.log(dists_sq)
    # Sum over upper triangle only
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(log_dists_sq[iu])
    
    # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / d_ij^2
    # But we need to project to tangent space
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ points
    grads = points * sum_weights[:, np.newaxis] - weighted_sum
    
    # Project to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points
    
    # Flatten gradient
    grad_flat = grads_tangent.flatten()
    
    return energy, grad_flat


def _optimize_lbfgsb(points, maxiter=2000, tol=1e-10):
    """Optimize using L-BFGS-B on the sphere."""
    x_flat = points.flatten()
    
    result = minimize(
        _energy_and_grad,
        x_flat,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': tol, 'maxfun': 50000}
    )
    
    opt_points = _project_to_sphere(result.x.reshape(N, 3))
    energy = _log_energy(opt_points)
    return opt_points, energy


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=42):
    """Generate random uniform points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _simulated_annealing(points, n_iter=2000, T_start=0.01, T_end=0.0001, seed=42):
    """Simulated annealing on sphere."""
    rng = np.random.RandomState(seed)
    current_points = points.copy()
    current_energy = _log_energy(current_points)
    best_points = current_points.copy()
    best_energy = current_energy
    
    for it in range(n_iter):
        # Temperature schedule
        t = it / max(n_iter - 1, 1)
        T = T_start * (T_end / T_start) ** t
        
        # Generate perturbation
        perturbation = rng.randn(N, 3) * T
        new_points = _project_to_sphere(current_points + perturbation)
        new_energy = _log_energy(new_points)
        
        delta = new_energy - current_energy
        if delta < 0 or rng.rand() < np.exp(-delta / max(T, 1e-15)):
            current_points = new_points
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = new_points.copy()
    
    return best_points, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_energy = float('inf')
    
    # Strategy 1: Fibonacci sphere + L-BFGS-B
    if time.time() < deadline - 60:
        points = _fibonacci_sphere(N)
        opt_points, energy = _optimize_lbfgsb(points, maxiter=3000, tol=1e-10)
        if energy < best_energy:
            best_energy = energy
            best_points = opt_points.copy()
    
    # Strategy 2: Multiple random starts with L-BFGS-B
    n_random_starts = 5
    for seed in range(n_random_starts):
        if time.time() > deadline - 20:
            break
        
        points = _random_uniform_sphere(N, seed=seed * 100 + 42)
        opt_points, energy = _optimize_lbfgsb(points, maxiter=2000, tol=1e-9)
        if energy < best_energy:
            best_energy = energy
            best_points = opt_points.copy()
    
    # Strategy 3: Simulated annealing from best solution
    if best_points is not None and time.time() < deadline - 15:
        sa_points, sa_energy = _simulated_annealing(
            best_points, n_iter=3000, T_start=0.005, T_end=0.0001, seed=123
        )
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_points = sa_points.copy()
        
        # Polish with L-BFGS-B after SA
        if time.time() < deadline - 5:
            opt_points, energy = _optimize_lbfgsb(best_points, maxiter=3000, tol=1e-11)
            if energy < best_energy:
                best_energy = energy
                best_points = opt_points.copy()
    
    # Final polish
    if best_points is not None and time.time() < deadline - 2:
        opt_points, energy = _optimize_lbfgsb(best_points, maxiter=5000, tol=1e-12)
        if energy < best_energy:
            best_energy = energy
            best_points = opt_points.copy()
    
    # Ensure valid output
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _fibonacci_sphere(N)
    
    best_points = _project_to_sphere(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
