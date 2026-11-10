"""Optimized logarithmic energy minimization for 150 points on the sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 150
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds before deadline to stop


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for N points on unit sphere.
    
    Energy: sum_{i<j} log(1/|x_i - x_j|)
    Gradient for point i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    # Pairwise distances
    # d[i,j] = |x_i - x_j|
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of log(1/dist_ij) = -sum over i<j of log(dist_ij)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each point i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j != i} diff[i,j] / dist_sq[i,j]
    # We need to sum over j for each i
    # diff is (n, n, 3), dist_sq is (n, n)
    # For each i, sum over j != i of diff[i,j] / dist_sq[i,j]
    # But dist_sq diagonal is set to 1, so we need to zero out diagonal contribution
    inv_dist_sq = np.zeros_like(dist_sq)
    np.fill_diagonal(inv_dist_sq, 0.0)
    np.fill_diagonal(dist_sq, 1.0)  # already 1
    inv_dist_sq = np.where(dist_sq > 1e-12, 1.0 / dist_sq, 0.0)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _golden_spiral():
    """Generate golden-angle Fibonacci spiral points on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points():
    """Generate random points on unit sphere."""
    pts = np.random.randn(N, 3)
    return _normalize(pts)


def _lbfgs_optimize(pts, time_limit):
    """Run L-BFGS-B optimization with time limit."""
    start_time = time.time()
    
    def objective_and_grad(flat_pts):
        pts = flat_pts.reshape(N, 3)
        e, g = _energy_and_grad(pts)
        return e, g.flatten()
    
    x0 = pts.flatten()
    
    result = minimize(
        objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': 10000,
            'ftol': 1e-14,
            'gtol': 1e-10,
            'maxls': 50,
        }
    )
    
    elapsed = time.time() - start_time
    pts_opt = _normalize(result.x.reshape(N, 3))
    return pts_opt, elapsed


def construct():
    """Main function: find 150 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    
    # Try multiple strategies and keep the best
    best_pts = _golden_spiral()
    best_energy = _energy_and_grad(best_pts)[0]
    
    # Strategy 1: L-BFGS from golden spiral
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        try:
            pts1, e1 = _lbfgs_optimize(best_pts, DEADLINE - SAFETY_MARGIN - elapsed)
            energy1 = _energy_and_grad(pts1)[0]
            if energy1 < best_energy:
                best_energy = energy1
                best_pts = pts1
        except Exception:
            pass
    
    # Strategy 2: L-BFGS from random points
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        try:
            random_seed = 42
            np.random.seed(random_seed)
            pts_rand = _random_points()
            pts2, e2 = _lbfgs_optimize(pts_rand, DEADLINE - SAFETY_MARGIN - elapsed)
            energy2 = _energy_and_grad(pts2)[0]
            if energy2 < best_energy:
                best_energy = energy2
                best_pts = pts2
        except Exception:
            pass
    
    # Strategy 3: Another random start with different seed
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        try:
            np.random.seed(123)
            pts_rand2 = _random_points()
            pts3, e3 = _lbfgs_optimize(pts_rand2, DEADLINE - SAFETY_MARGIN - elapsed)
            energy3 = _energy_and_grad(pts3)[0]
            if energy3 < best_energy:
                best_energy = energy3
                best_pts = pts3
        except Exception:
            pass
    
    # Final polish: very tight L-BFGS on best result
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        try:
            def objective_and_grad_tight(flat_pts):
                pts = flat_pts.reshape(N, 3)
                e, g = _energy_and_grad(pts)
                return e, g.flatten()
            
            x0 = best_pts.flatten()
            result = minimize(
                objective_and_grad_tight,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 5000,
                    'ftol': 1e-15,
                    'gtol': 1e-12,
                    'maxls': 50,
                }
            )
            pts_final = _normalize(result.x.reshape(N, 3))
            energy_final = _energy_and_grad(pts_final)[0]
            if energy_final < best_energy:
                best_energy = energy_final
                best_pts = pts_final
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = _normalize(best_pts)
    
    # Check for non-finite values or too-close points
    if not np.all(np.isfinite(best_pts)):
        best_pts = _golden_spiral()
    
    # Check minimum distance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    if min_dist < 1e-9:
        # Fallback to golden spiral
        best_pts = _golden_spiral()
    
    return best_pts


if __name__ == "__main__":
    pts = construct()
    energy = _energy_and_grad(pts)[0]
    print(f"Energy: {energy}")
    print(f"Shape: {pts.shape}")
