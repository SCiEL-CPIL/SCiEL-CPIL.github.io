"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 100
DEADLINE = 150.0  # wall-clock limit from evaluator
SAFETY_MARGIN = 10.0  # seconds to leave before deadline


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    # Compute pairwise distances
    # Use efficient computation
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask diagonal and upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Avoid log(0) - but points should be distinct
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d))


def _project_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate n random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _linalg_optimize(points, max_iter=500, tol=1e-12):
    """
    Optimize points on sphere using gradient descent with projection.
    Uses the fact that for points on sphere, we can use spherical coordinates
    or simply project back after Euclidean gradient steps.
    """
    n = len(points)
    points = _project_to_sphere(points.copy())
    
    best_energy = _log_energy(points)
    best_points = points.copy()
    
    # Gradient of energy w.r.t. positions (in R^3), then project to tangent space
    lr = 0.01
    for it in range(max_iter):
        # Compute forces (negative gradient of energy)
        # Energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
        # dE/dx_i = -sum_{j!=i} (1/d_ij) * (x_i - x_j)/d_ij = -sum_{j!=i} (x_i - x_j)/d_ij^2
        # Wait: d/dx_i log(1/d_ij) = d/dx_i (-log d_ij) = -(1/d_ij) * d(d_ij)/d(x_i)
        # d(d_ij)/d(x_i) = (x_i - x_j)/d_ij
        # So d/dx_i log(1/d_ij) = -(x_i - x_j)/d_ij^2
        # Total gradient for x_i: sum_{j!=i} -(x_i - x_j)/d_ij^2 = sum_{j!=i} (x_j - x_i)/d_ij^2
        
        # Compute pairwise differences and distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # diff[i,j] = x_i - x_j
        dist_sq = np.sum(diff * diff, axis=2)
        dist = np.sqrt(np.maximum(dist_sq, 1e-30))
        
        # Avoid division by zero on diagonal
        np.fill_diagonal(dist, 1.0)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Gradient: grad[i] = sum_{j!=i} (x_j - x_i) / d_ij^2
        # = -sum_{j!=i} (x_i - x_j) / d_ij^2
        # Using diff[i,j] = x_i - x_j, so x_j - x_i = -diff[i,j]
        # grad[i] = sum_{j!=i} -diff[i,j] / dist_sq[i,j]
        
        # weight matrix: W[i,j] = 1/dist_sq[i,j] for i!=j, 0 for i==j
        W = 1.0 / dist_sq
        np.fill_diagonal(W, 0.0)
        
        # grad[i] = -sum_j W[i,j] * diff[i,j]
        # diff[i,j] = x_i - x_j, so sum_j W[i,j]*diff[i,j] = sum_j W[i,j]*(x_i - x_j)
        # = x_i * sum_j W[i,j] - sum_j W[i,j]*x_j
        
        Wsum = W.sum(axis=1, keepdims=True)  # shape (n,1)
        # sum_j W[i,j]*x_j = W @ points
        Wx = W @ points  # shape (n,3)
        
        # grad[i] = -(x_i * Wsum[i] - Wx[i]) = Wx[i] - x_i * Wsum[i]
        grad = Wx - points * Wsum  # shape (n,3)
        
        # Project gradient to tangent space: grad_tangent = grad - (grad·x_i)*x_i
        dot = np.sum(grad * points, axis=1, keepdims=True)
        grad_tangent = grad - dot * points
        
        # Update
        points = points + lr * grad_tangent
        points = _project_to_sphere(points)
        
        # Adaptive learning rate
        new_energy = _log_energy(points)
        if new_energy < best_energy:
            best_energy = new_energy
            best_points = points.copy()
            # Can increase lr slightly
            lr = min(lr * 1.05, 0.1)
        else:
            # Roll back and decrease lr
            points = best_points.copy()
            lr *= 0.5
            if lr < 1e-8:
                break
    
    return best_points, best_energy


def _scipy_optimize(points, max_iter=1000):
    """Use scipy optimize with sphere constraint via Lagrange multipliers."""
    from scipy.optimize import minimize
    
    n = len(points)
    
    def objective_and_grad(flat):
        pts = flat.reshape(n, 3)
        pts = _project_to_sphere(pts)
        
        # Compute energy
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist = np.sqrt(np.maximum(dist_sq, 1e-30))
        np.fill_diagonal(dist, 1.0)
        np.fill_diagonal(dist_sq, 1.0)
        
        W = 1.0 / dist_sq
        np.fill_diagonal(W, 0.0)
        
        Wsum = W.sum(axis=1, keepdims=True)
        Wx = W @ pts
        grad = Wx - pts * Wsum
        
        # Project to tangent
        dot = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tangent = grad - dot * pts
        
        # Energy
        iu = np.triu_indices(n, k=1)
        d = dist[iu]
        d = np.maximum(d, 1e-15)
        energy = math.fsum(np.log(1.0 / d))
        
        return energy, grad_tangent.flatten()
    
    # Use L-BFGS-B on the manifold by parameterizing with tangent vectors
    # Actually, let's just use the projected gradient approach with scipy
    # We'll use a simple approach: optimize in R^3 with penalty for non-unit norm
    
    def penalty_objective(flat):
        pts = flat.reshape(n, 3)
        # Penalty for not being on sphere
        norms_sq = np.sum(pts * pts, axis=1)
        penalty = 1000.0 * np.sum((norms_sq - 1.0) ** 2)
        
        pts = _project_to_sphere(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist = np.sqrt(np.maximum(dist_sq, 1e-30))
        np.fill_diagonal(dist, 1.0)
        iu = np.triu_indices(n, k=1)
        d = dist[iu]
        d = np.maximum(d, 1e-15)
        energy = math.fsum(np.log(1.0 / d))
        return energy + penalty
    
    x0 = points.flatten()
    result = minimize(penalty_objective, x0, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-14})
    
    best_points = _project_to_sphere(result.x.reshape(n, 3))
    best_energy = _log_energy(best_points)
    return best_points, best_energy


def construct():
    start_time = time.time()
    
    # Start with Fibonacci spiral
    best_points = _fibonacci_sphere(N)
    best_energy = _log_energy(best_points)
    
    # Try several random restarts with optimization
    for seed in range(20):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Random start
        pts = _random_sphere(N, seed=seed)
        
        # First do gradient descent optimization
        try:
            opt_pts, opt_energy = _linalg_optimize(pts, max_iter=200, tol=1e-12)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_points = opt_pts.copy()
        except Exception:
            pass
        
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Then try scipy optimization from this result
        try:
            scipy_pts, scipy_energy = _scipy_optimize(opt_pts, max_iter=200)
            if scipy_energy < best_energy:
                best_energy = scipy_energy
                best_points = scipy_pts.copy()
        except Exception:
            pass
    
    # Final polish with more iterations
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        try:
            final_pts, final_energy = _linalg_optimize(best_points, max_iter=500, tol=1e-14)
            if final_energy < best_energy:
                best_energy = final_energy
                best_points = final_pts.copy()
        except Exception:
            pass
    
    # Ensure points are valid
    best_points = _project_to_sphere(best_points)
    
    # Check minimum distance
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = dist[iu].min()
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci if points are too close
        best_points = _fibonacci_sphere(N)
    
    return best_points


# EVOLVE-BLOCK-START
# (The actual construct function is defined above)
# EVOLVE-BLOCK-END
