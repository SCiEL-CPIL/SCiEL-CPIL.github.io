"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time
import math
import numpy as np
from scipy.optimize import minimize, L_BFGS_B

DEADLINE_MARGIN = 10.0  # seconds to leave before 150s kill


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1 / |x_i - x_j|)."""
    n = pts.shape[0]
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Mask diagonal and upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Clip to avoid log(0)
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())


def _energy_and_grad_flat(x, N):
    """Energy and gradient for flat vector x of length 3*N."""
    pts = x.reshape(N, 3)
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist2, 1.0)  # avoid division by zero
    dist = np.sqrt(dist2)
    # log(1/d) = -log(d), gradient w.r.t. pts_i: sum_{j!=i} (x_i - x_j)/|x_i-x_j|^2 * ... 
    # d/dx_i log(1/|x_i - x_j|) = -d/dx_i log|x_i - x_j| = -(x_i - x_j)/|x_i - x_j|^2
    # So gradient of energy w.r.t. x_i = sum_{j != i} -(x_i - x_j)/|x_i - x_j|^2
    # = sum_{j != i} (x_j - x_i)/|x_i - x_j|^2
    # inv_dist2 = 1/dist^2
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    # grad[i] = sum_j inv_dist2[i,j] * (x_j - x_i) = -sum_j inv_dist2[i,j] * (x_i - x_j)
    # = -inv_dist2 @ (x_i - x_j) summed over j... 
    # More carefully: grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    # = sum_j (x_j - x_i) * inv_dist2[i,j]
    # = (sum_j x_j * inv_dist2[i,j]) - x_i * sum_j inv_dist2[i,j]
    sum_inv = inv_dist2.sum(axis=1)
    grad = pts * sum_inv[:, np.newaxis] - (inv_dist2 @ pts)
    # Note: inv_dist2 @ pts gives sum_j inv_dist2[i,j] * x_j
    # So grad = pts * sum_inv - (inv_dist2 @ pts) = sum_j inv_dist2[i,j]*(x_i - x_j) = -gradient of energy
    # Wait, let me re-derive.
    # Energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|x_i - x_j|
    # dE/dx_i = -sum_{j != i} (1/2) * (x_i - x_j)/|x_i - x_j|^2 * 2 ... 
    # Actually d/dx_i log|x_i - x_j| = (x_i - x_j)/|x_i - x_j|^2
    # So dE/dx_i = -sum_{j != i, j != i} (x_i - x_j)/|x_i - x_j|^2 for the pairs involving i
    # For pair (i,j) with i<j: contributes to dE/dx_i: -(x_i-x_j)/|x_i-x_j|^2 and to dE/dx_j: -(x_j-x_i)/|x_j-x_i|^2 = (x_i-x_j)/|x_i-x_j|^2
    # So dE/dx_i = -sum_{j != i} (x_i - x_j)/|x_i - x_j|^2 = sum_{j != i} (x_j - x_i)/|x_i - x_j|^2
    # My formula: grad = pts * sum_inv - (inv_dist2 @ pts)
    # = x_i * sum_j inv_dist2[i,j] - sum_j inv_dist2[i,j] * x_j
    # = sum_j inv_dist2[i,j] * (x_i - x_j)
    # = sum_{j != i} (x_i - x_j)/|x_i - x_j|^2
    # This is -dE/dx_i. So the actual gradient is -grad.
    grad = -grad
    energy = _log_energy(pts)
    return energy, grad.ravel()


def _project_to_sphere(x, N):
    """Project flat vector to unit sphere."""
    pts = x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_sphere(N, offset=0.5):
    i = np.arange(N) + offset
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    t_start = time.time()
    deadline = t_start + 150.0 - DEADLINE_MARGIN
    
    # Best result so far
    best_pts = _fibonacci_sphere(N)
    best_energy = _log_energy(best_pts)
    
    def time_left():
        return deadline - time.time()
    
    # Strategy: multiple random restarts with gradient descent on sphere
    # Use scipy L-BFGS-B on the flat representation, projecting to sphere periodically
    
    rng = np.random.default_rng(42)
    
    def optimize_from(start_pts, max_iter=2000, tol=1e-12):
        """Optimize starting from start_pts using gradient descent on sphere."""
        x = start_pts.ravel().copy()
        n = N * 3
        
        # Use L-BFGS-B with custom gradient
        # We need to handle the sphere constraint. We'll use a penalty-free approach:
        # optimize in R^{3N} but re-project to sphere after each iteration.
        # Actually, let's use a simpler approach: gradient descent with line search on the sphere.
        
        # Alternative: use scipy minimize with the flat parameterization and 
        # normalize at the end. This isn't ideal but works reasonably.
        
        # Better: use the fact that the energy is invariant under scaling of each point,
        # so we can just optimize in R^{3N} and normalize at the end.
        
        def func_grad(x_flat):
            pts = x_flat.reshape(N, 3)
            # Normalize
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            pts_norm = pts / norms
            energy = _log_energy(pts_norm)
            # Gradient on the sphere: project the Euclidean gradient to the tangent space
            _, euclidean_grad = _energy_and_grad_flat(x_flat, N)
            grad = euclidean_grad.reshape(N, 3)
            # Project to tangent space: grad_tangent = grad - (grad . x) x
            dots = np.einsum('ij,ij->i', grad, pts_norm)
            grad_tangent = grad - dots[:, np.newaxis] * pts_norm
            return energy, grad_tangent.ravel()
        
        # Use L-BFGS-B
        result = minimize(
            func_grad,
            x,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10, 'maxfun': max_iter * 2}
        )
        
        # Normalize the result
        best_x = result.x.reshape(N, 3)
        norms = np.linalg.norm(best_x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        best_x = best_x / norms
        return best_x, result.fun
    
    # Multiple restarts
    n_restarts = 20
    for r in range(n_restarts):
        if time_left() < 30:  # Need at least 30s for one optimization
            break
        
        if r == 0:
            start = best_pts.copy()
        else:
            # Random start with some structure
            # Mix of random and Fibonacci-based perturbations
            if r % 3 == 0:
                # Random uniform on sphere
                start = rng.normal(size=(N, 3))
                norms = np.linalg.norm(start, axis=1, keepdims=True)
                start = start / np.maximum(norms, 1e-15)
            elif r % 3 == 1:
                # Fibonacci with random perturbation
                start = _fibonacci_sphere(N)
                perturbation = rng.normal(scale=0.1, size=(N, 3))
                start = start + perturbation
                norms = np.linalg.norm(start, axis=1, keepdims=True)
                start = start / np.maximum(norms, 1e-15)
            else:
                # Random with some clustering
                start = rng.normal(size=(N, 3))
                # Add some structure by mixing with Fibonacci
                fib = _fibonacci_sphere(N)
                alpha = 0.3 + 0.4 * rng.random()
                start = alpha * start + (1 - alpha) * fib
                norms = np.linalg.norm(start, axis=1, keepdims=True)
                start = start / np.maximum(norms, 1e-15)
        
        try:
            opt_pts, opt_energy = optimize_from(start, max_iter=1000)
            if np.all(np.isfinite(opt_pts)):
                # Verify energy
                actual_energy = _log_energy(opt_pts)
                if actual_energy < best_energy:
                    best_pts = opt_pts
                    best_energy = actual_energy
        except Exception:
            pass
    
    # Final polish: more iterations on the best candidate
    if time_left() > 15:
        try:
            polished_pts, polished_energy = optimize_from(best_pts, max_iter=5000, tol=1e-14)
            if np.all(np.isfinite(polished_pts)):
                actual_energy = _log_energy(polished_pts)
                if actual_energy < best_energy:
                    best_pts = polished_pts
                    best_energy = actual_energy
        except Exception:
            pass
    
    # One more polish if time allows
    if time_left() > 10:
        try:
            polished_pts, polished_energy = optimize_from(best_pts, max_iter=3000, tol=1e-15)
            if np.all(np.isfinite(polished_pts)):
                actual_energy = _log_energy(polished_pts)
                if actual_energy < best_energy:
                    best_pts = polished_pts
                    best_energy = actual_energy
        except Exception:
            pass
    
    # Final validation
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_sphere(N)
    
    # Ensure no points are too close
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts
