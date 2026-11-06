"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import minimize


def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, 1)
    d = dist[iu]
    d = np.maximum(d, 1e-15)
    return math.fsum(math.log(1.0 / di) for di in d)


def _log_energy_and_grad(points):
    """Compute logarithmic energy and its gradient (projected to tangent space)."""
    # Vectorized pairwise computation
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 3)
    d2 = np.einsum('ijk,ijk->ij', diff, diff)  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(d2, 1.0)
    
    # Energy: sum_{i<j} log(1/sqrt(d2_ij)) = -0.5 * sum_{i<j} log(d2_ij)
    iu = np.triu_indices(N, 1)
    d2_upper = d2[iu]
    d2_upper = np.maximum(d2_upper, 1e-30)
    energy = -0.5 * math.fsum(math.log(di) for di in d2_upper)
    
    # Gradient: dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Because d/dx_i [log(1/|x_i - x_j|)] = -(x_i - x_j)/|x_i - x_j|^2
    # So dE/dx_i = -sum_{j != i} (x_i - x_j)/|x_i - x_j|^2 = sum_{j != i} (x_j - x_i)/|x_i - x_j|^2
    # Wait, let me re-derive:
    # E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|x_i - x_j|
    # dE/dx_i = -sum_{j != i} d/dx_i log|x_i - x_j| = -sum_{j != i} (x_i - x_j)/|x_i - x_j|^2
    # So grad_i = -sum_{j != i} (x_i - x_j)/d2_ij = sum_{j != i} (x_j - x_i)/d2_ij
    
    # Vectorized: for each i, grad_i = sum_{j != i} (x_j - x_i) / d2_ij
    # = sum_j (x_j - x_i) / d2_ij - (x_i - x_i)/d2_ii = sum_j (x_j - x_i) / d2_ij
    # We can compute: grad_i = sum_j x_j / d2_ij - x_i * sum_j 1/d2_ij
    inv_d2 = 1.0 / d2  # (N, N)
    # For each i: sum_j x_j / d2_ij
    term1 = np.einsum('ij,jk->ik', inv_d2, points)  # (N, 3)
    # For each i: x_i * sum_j 1/d2_ij
    sum_inv = np.sum(inv_d2, axis=1)  # (N,)
    term2 = points * sum_inv[:, np.newaxis]
    grad = term1 - term2
    
    # Project gradient to tangent space: grad_i -= (grad_i . x_i) * x_i
    dot = np.einsum('ij,ij->i', grad, points)
    grad = grad - dot[:, np.newaxis] * points
    
    return energy, grad


def _project_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _fibonacci_spiral(n, seed_offset=0.0):
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + seed_offset
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    """
    Multi-start gradient-based optimization for logarithmic energy minimization.
    Uses vectorized gradient computation and multiple random initializations.
    """
    best_points = None
    best_energy = float('inf')
    
    np.random.seed(42)
    
    # Define objective and gradient for optimizer
    def objective_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        pts = _project_to_sphere(pts)
        e, g = _log_energy_and_grad(pts)
        return e, g.flatten()
    
    # Generate diverse initial configurations
    n_trials = 8
    for trial in range(n_trials):
        if trial == 0:
            x0 = _fibonacci_spiral(N)
        elif trial == 1:
            x0 = _fibonacci_spiral(N, seed_offset=0.3)
        elif trial == 2:
            x0 = _fibonacci_spiral(N, seed_offset=0.7)
        elif trial == 3:
            # Random points
            x0 = np.random.randn(N, 3)
            x0 = _project_to_sphere(x0)
        elif trial == 4:
            # Fibonacci with moderate perturbation
            x0 = _fibonacci_spiral(N)
            x0 += np.random.normal(0, 0.1, size=(N, 3))
            x0 = _project_to_sphere(x0)
        elif trial == 5:
            # Fibonacci with larger perturbation
            x0 = _fibonacci_spiral(N)
            x0 += np.random.normal(0, 0.3, size=(N, 3))
            x0 = _project_to_sphere(x0)
        elif trial == 6:
            # Two-cluster initialization
            center1 = np.array([1.0, 0, 0])
            center2 = np.array([-1.0, 0, 0])
            x0 = np.zeros((N, 3))
            for i in range(N):
                if i % 2 == 0:
                    x0[i] = center1 + np.random.randn(3) * 0.3
                else:
                    x0[i] = center2 + np.random.randn(3) * 0.3
            x0 = _project_to_sphere(x0)
        else:
            # Random with Fibonacci blend
            x0 = _fibonacci_spiral(N)
            x0 = 0.5 * x0 + 0.5 * np.random.randn(N, 3)
            x0 = _project_to_sphere(x0)
        
        x_flat = x0.flatten()
        
        # Run L-BFGS-B optimization
        try:
            res = minimize(
                objective_and_grad,
                x_flat,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'gtol': 1e-9, 'ftol': 1e-12, 'maxls': 50}
            )
            pts = _project_to_sphere(res.x.reshape(N, 3))
            e, _ = _log_energy_and_grad(pts)
            
            if e < best_energy:
                best_energy = e
                best_points = pts.copy()
        except Exception:
            continue
    
    # Refinement pass on the best solution
    if best_points is not None:
        x_flat = best_points.flatten()
        try:
            res = minimize(
                objective_and_grad,
                x_flat,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14, 'maxls': 50}
            )
            pts = _project_to_sphere(res.x.reshape(N, 3))
            e, _ = _log_energy_and_grad(pts)
            if e < best_energy:
                best_energy = e
                best_points = pts.copy()
        except Exception:
            pass
    
    # Fallback to Fibonacci spiral
    if best_points is None:
        best_points = _fibonacci_spiral(N)
    
    return best_points
# EVOLVE-BLOCK-END
