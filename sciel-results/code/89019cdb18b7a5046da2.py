"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math

def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    # points: (N, 3) array
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, 1)
    d = dist[iu]
    d = np.maximum(d, 1e-15)
    return math.fsum(math.log(1.0 / di) for di in d)

def _tangent_gradient(points):
    """
    Compute the gradient of the energy projected onto the tangent space of the sphere.
    Energy E = sum_{i<j} log(1/|x_i - x_j|) = - sum_{i<j} log(|x_i - x_j|)
    Gradient wrt x_i: sum_{j!=i} - (x_i - x_j) / |x_i - x_j|^2
    We want to minimize E, so we move in direction -grad_E.
    But standard GD updates x <- x - lr * grad_E.
    So we return grad_E projected to tangent.
    """
    # diff[i, j] = x_i - x_j
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # inv_dist_sq = 1 / |x_i - x_j|^2
    inv_dist_sq = 1.0 / dist_sq
    
    # grad[i] = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # This is sum_j (diff[i,j] * inv_dist_sq[i,j])
    # Note: diagonal term is 0 because diff[i,i]=0
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Project to tangent space: grad_tangent = grad - (grad . x_i) x_i
    proj = np.sum(grad * points, axis=1, keepdims=True)
    grad_tangent = grad - proj * points
    
    return grad_tangent

def _project_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def construct():
    """
    Use Gradient Descent with Nesterov Momentum on the sphere.
    """
    best_points = None
    best_energy = float('inf')
    
    # Fibonacci spiral initialization
    def get_fibonacci_spiral(noise_scale=0.0):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        if noise_scale > 0:
            pts += np.random.normal(0, noise_scale, size=(N, 3))
            pts = _project_to_sphere(pts)
        return pts

    np.random.seed(42)
    
    # Multi-start with different noise levels
    # Start with low noise to converge fast, then higher noise to explore
    noise_scales = [0.0, 0.05, 0.1, 0.2]
    
    for noise in noise_scales:
        x = get_fibonacci_spiral(noise)
        
        # Nesterov Momentum parameters
        v = np.zeros_like(x)
        lr = 0.05
        momentum = 0.9
        max_iters = 3000
        
        # Adaptive learning rate / Line search
        # We will use a simple backtracking line search for stability
        
        for it in range(max_iters):
            grad = _tangent_gradient(x)
            
            # Nesterov step
            v = momentum * v - lr * grad
            x_new = x + v
            x_new = _project_to_sphere(x_new)
            
            # Check convergence / step size
            # If energy increased significantly, reduce lr (simple heuristic)
            # For speed, we skip full line search and rely on small lr + momentum
            # But we must ensure we don't diverge.
            # Let's do a quick check every 100 steps or if energy is bad
            if it % 10 == 0:
                e_new = _log_energy(x_new)
                if not math.isfinite(e_new) or e_new > best_energy + 100:
                    # Reset or reduce lr
                    lr *= 0.5
                    v = np.zeros_like(x)
                    x_new = x
                    continue
                
            x = x_new
            
            # Reduce learning rate slowly to allow fine-tuning
            if it > 1000:
                lr *= 0.999
            
            # Early stopping if gradient norm is small
            if it > 100 and np.linalg.norm(grad) < 1e-6:
                break
        
        e_final = _log_energy(x)
        if e_final < best_energy:
            best_energy = e_final
            best_points = x
    
    # If no valid points found (should not happen), fallback
    if best_points is None:
        best_points = get_fibonacci_spiral(0.0)
        
    return best_points
# EVOLVE-BLOCK-END
