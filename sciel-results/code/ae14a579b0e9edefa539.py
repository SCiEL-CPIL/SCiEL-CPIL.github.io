"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 150
TIME_LIMIT = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _log_energy_and_grad(points):
    """
    Compute the logarithmic energy and its gradient on the sphere.
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient: for each point x_i, grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    # Compute pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dists_sq = np.sum(diff * diff, axis=-1)  # (n, n)
    np.fill_diagonal(dists_sq, 1.0)  # avoid division by zero
    
    # Energy: -sum_{i<j} log(|x_i - x_j|)
    # = -0.5 * sum_{i != j} log(|x_i - x_j|)
    dists = np.sqrt(dists_sq)
    log_dists = np.log(dists)
    np.fill_diagonal(log_dists, 0.0)
    energy = -0.5 * np.sum(log_dists)
    
    # Gradient: grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Note: diff[i,j] = x_i - x_j, so we need (x_i - x_j) / |x_i - x_j|^2
    # = diff / dists_sq
    grad = -np.sum(diff / dists_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


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


def _optimize_from(points, deadline, rng, maxiter=200):
    """
    Optimize points using gradient-based Riemannian optimization.
    We project the gradient onto the tangent space and use L-BFGS-B.
    """
    start_time = time.time()
    
    def objective_and_grad(x):
        pts = _normalize(x.reshape(-1, 3))
        energy, grad = _log_energy_and_grad(pts)
        
        # Project gradient to tangent space: grad_tangent = grad - (grad . x_i) * x_i
        grad_tangent = np.zeros_like(grad)
        for i in range(len(pts)):
            proj = np.dot(grad[i], pts[i])
            grad_tangent[i] = grad[i] - proj * pts[i]
        
        # Flatten gradient
        grad_flat = grad_tangent.flatten()
        
        return energy, grad_flat
    
    # Initial evaluation
    energy, grad = objective_and_grad(points.flatten())
    
    # Use L-BFGS-B with projection back to sphere
    result = minimize(
        objective_and_grad,
        points.flatten(),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': maxiter, 'maxls': 50}
    )
    
    # Normalize the result
    result_points = _normalize(result.x.reshape(-1, 3))
    
    # Check time
    if time.time() > deadline - 5:
        return result_points
    
    # Evaluate final energy
    final_energy, _ = _log_energy_and_grad(result_points)
    
    return result_points


def _kick(points, rng, strength=0.1):
    """
    Apply a random kick to some points to escape local minima.
    """
    n = len(points)
    # Randomly select 10-20% of points to kick
    n_kick = max(1, int(0.15 * n))
    idx = rng.choice(n, size=n_kick, replace=False)
    
    kicked = points.copy()
    for i in idx:
        # Add random perturbation
        noise = rng.standard_normal(3) * strength
        kicked[i] = points[i] + noise
        kicked[i] = _normalize(kicked[i][:np.newaxis, :])[0]
    
    return kicked


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_energy = float('inf')
    best_points = _fibonacci_sphere(N)

    # 1. Evaluate initial Fibonacci
    try:
        e, _ = _log_energy_and_grad(best_points)
        if e < best_energy:
            best_energy = e
    except Exception:
        pass

    # 2. Refine the Fibonacci start with gradient optimization
    rng = np.random.default_rng(42)
    try:
        if time.time() < deadline - 20:
            refined = _optimize_from(best_points, deadline, rng, maxiter=100)
            e, _ = _log_energy_and_grad(refined)
            if e < best_energy:
                best_energy = e
                best_points = refined
    except Exception:
        pass

    # 3. Portfolio of random restarts with gradient optimization and kicks
    num_restarts = 50
    
    for k in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random start
        pts = _random_sphere(N, rng)
        
        # Quick gradient refinement
        try:
            if time.time() < deadline - 15:
                refined_pts = _optimize_from(pts, deadline, rng, maxiter=50)
                e, _ = _log_energy_and_grad(refined_pts)

                if e < best_energy:
                    best_energy = e
                    best_points = refined_pts

                    # If it's good, refine more
                    if time.time() < deadline - 20:
                        further_refined = _optimize_from(refined_pts, deadline, rng, maxiter=150)
                        e2, _ = _log_energy_and_grad(further_refined)
                        if e2 < best_energy:
                            best_energy = e2
                            best_points = further_refined
        except Exception:
            continue
        
        # Try kick-restart from best
        if time.time() < deadline - 25 and k % 5 == 0:
            try:
                kicked_pts = _kick(best_points, rng, strength=0.05 + 0.02 * (k // 5))
                refined_kicked = _optimize_from(kicked_pts, deadline, rng, maxiter=100)
                e3, _ = _log_energy_and_grad(refined_kicked)
                if e3 < best_energy:
                    best_energy = e3
                    best_points = refined_kicked
            except Exception:
                pass

    # 4. Final polish if time permits
    if time.time() < deadline - 5:
        try:
            final_pts = _optimize_from(best_points, deadline, rng, maxiter=300)
            e, _ = _log_energy_and_grad(final_pts)
            if e < best_energy:
                best_energy = e
                best_points = final_pts
        except Exception:
            pass

    return _normalize(best_points)

# EVOLVE-BLOCK-END
