"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math

N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|p_i - p_j|)."""
    # points: (N, 3) array on unit sphere
    # Compute pairwise distances
    # Use broadcasting for efficiency
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Zero out diagonal
    np.fill_diagonal(dists, 1.0)
    # Avoid log(0) or negative issues; distances should be > 0
    # Clip to avoid log of values <= 0
    dists = np.maximum(dists, 1e-15)
    # Compute log(1/d) = -log(d)
    log_terms = -np.log(dists)
    # Sum over upper triangle (i < j)
    # Use fsum for accuracy
    idx = np.triu_indices(N, k=1)
    vals = log_terms[idx]
    return math.fsum(vals.tolist())


def _log_energy_grad(points):
    """Compute gradient of logarithmic energy w.r.t. points on R^3.
    
    E = sum_{i<j} log(1/|p_i - p_j|) = -sum_{i<j} log|p_i - p_j|
    dE/dp_k = sum_{j != k} (p_k - p_j) / |p_k - p_j|^2
    """
    # points: (N, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 3)
    dists_sq = np.sum(diff * diff, axis=2)  # (N, N)
    # Avoid division by zero on diagonal
    dists_sq = np.where(dists_sq > 0, dists_sq, 1.0)
    # grad_k = sum_{j != k} (p_k - p_j) / |p_k - p_j|^2
    # diff[k,j] = p_k - p_j
    # So grad[k] = sum_j diff[k,j] / dists_sq[k,j] for j != k
    inv_dists_sq = np.zeros_like(dists_sq)
    np.fill_diagonal(inv_dists_sq, 0.0)
    np.divide(1.0, dists_sq, out=inv_dists_sq, where=(dists_sq > 0))
    grad = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    return grad


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def construct():
    # Start with golden-angle spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    x0 = _project_to_sphere(x0)
    
    # Use L-BFGS-B with a penalty for points not on sphere
    # Alternatively, we can use a constrained optimization approach
    
    # Let's use scipy's minimize with BFGS and a custom constraint
    # But since we need points on the sphere, let's use a different approach:
    # Parameterize each point by its direction and optimize on R^3 with projection
    
    # Actually, let's try a simpler approach: use the gradient projected onto the tangent space
    # and do a manual Riemannian gradient descent
    
    # Or better: use scipy.optimize.minimize with bounds/penalty
    
    # Let's try using the fact that the energy is smooth on the sphere
    # We'll use a line search gradient descent with projection
    
    # First, let's try scipy's minimize with a penalty method
    def objective(x_flat):
        pts = x_flat.reshape(N, 3)
        pts = _project_to_sphere(pts)
        return _log_energy(pts)
    
    def gradient(x_flat):
        pts = x_flat.reshape(N, 3)
        pts = _project_to_sphere(pts)
        grad = _log_energy_grad(pts)
        # Project gradient onto tangent space
        # Tangent projection: g - (g . p) * p
        dot = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tangent = grad - dot * pts
        return grad_tangent.flatten()
    
    # Use L-BFGS-B on the flattened representation
    # But this doesn't enforce the sphere constraint directly
    # Let's instead use a manifold-aware approach
    
    # Manual Riemannian gradient descent with backtracking line search
    x = x0.copy()
    lr = 0.01
    min_lr = 1e-8
    max_iter = 5000
    tol = 1e-12
    
    for it in range(max_iter):
        grad = _log_energy_grad(x)
        # Project to tangent space
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad_tangent = grad - dot * x
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < tol:
            break
        
        # Backtracking line search
        step = lr
        current_energy = _log_energy(x)
        
        for _ in range(50):
            # Riemannian exponential map (approximation for small steps)
            # For sphere: exp_p(v) = p*cos(|v|) + v/|v|*sin(|v|)
            # For small steps, this is approximately p + v
            # Let's use the exact formula
            v = step * grad_tangent
            v_norms = np.linalg.norm(v, axis=1, keepdims=True)
            v_norms_safe = np.maximum(v_norms, 1e-15)
            
            # Riemannian exponential map
            new_x = x * np.cos(v_norms_safe) + (v / v_norms_safe) * np.sin(v_norms_safe)
            # Handle where v_norms was very small
            small = v_norms < 1e-15
            if np.any(small):
                new_x[small] = x[small] + v[small]
            
            # Project to ensure unit norm (numerical safety)
            new_x = _project_to_sphere(new_x)
            
            new_energy = _log_energy(new_x)
            
            if new_energy < current_energy - 1e-4 * step * grad_norm:
                x = new_x
                current_energy = new_energy
                break
            else:
                step *= 0.5
        
        # Adapt learning rate
        if step < min_lr:
            break
    
    # Final projection
    x = _project_to_sphere(x)
    
    # Verify all points are finite and distinct
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(x[i] - x[j])
            if d < 1e-9:
                # Nudge slightly
                x[i] += 1e-8 * np.random.randn(3)
                x[i] = _project_to_sphere(x[i])
    
    return x

# EVOLVE-BLOCK-END
