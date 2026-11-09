"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time

def _energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/d_ij)."""
    # pts: (N, 3)
    # Compute all pairwise distances
    # d_ij = ||x_i - x_j||
    # We need sum over i<j of log(1/d_ij) = -sum log(d_ij)
    # Use vectorized computation
    # d^2 = (xi - xj)^2
    # To avoid memory blowup for large N, we can do it in chunks, but N=150 is small
    # 150*150 = 22500 entries, fine.
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    d2 = np.sum(diff * diff, axis=2)  # (N, N)
    # Mask the diagonal and lower triangle
    # We only want i < j
    iu = np.triu_indices(N, k=1)
    d2_upper = d2[iu]
    # Clip to avoid log(0) or negative due to float errors
    d2_upper = np.maximum(d2_upper, 1e-30)
    d_upper = np.sqrt(d2_upper)
    # Energy = sum log(1/d) = -sum log(d)
    e = -np.sum(np.log(d_upper))
    return e

def _gradient(pts):
    """Compute gradient of energy w.r.t. points, projected to tangent space."""
    # E = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j != i} (x_i - x_j) / d_ij^2
    # Wait, let's derive carefully.
    # E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # d/dx_i log(d_ij) = (1/d_ij) * d/dx_i d_ij = (1/d_ij) * (x_i - x_j)/d_ij = (x_i - x_j)/d_ij^2
    # So dE/dx_i = -sum_{j != i} (x_i - x_j)/d_ij^2
    # Note: for each pair (i,j), the contribution to grad_i is -(x_i-x_j)/d^2 and to grad_j is +(x_i-x_j)/d^2
    # Let's compute pairwise forces.
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    d2 = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(d2, 1.0)  # Avoid division by zero
    # inv_d2 = 1/d2
    inv_d2 = 1.0 / d2  # (N, N)
    # grad_i = -sum_{j != i} (x_i - x_j) / d_ij^2
    # diff[i,j] = x_i - x_j
    # So grad = -sum_j diff[i,j] * inv_d2[i,j]
    # But we need to exclude j=i. Since we set d2[i,i]=1, diff[i,i]=0, so it's automatically 0.
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)  # (N, 3)
    
    # Project gradient to tangent space: g_t = g - (g . x) x
    # Since points are on sphere, x . x = 1
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tangent = grad - dot * pts
    return grad_tangent

def _project_to_sphere(pts):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Gradient Descent with adaptive step size
    # Use a simple line search or fixed step with decay
    # Let's try a combination of gradient descent and maybe some random restarts if time permits?
    # No, stick to one good trajectory to maximize quality.
    
    # We can use a simple adaptive step size.
    # Start with a large step, reduce if energy increases.
    step = 0.01
    best_pts = pts.copy()
    best_energy = _energy(best_pts)
    
    # Pre-compute something? No.
    
    for it in range(20000):
        if time.time() > deadline:
            break
        
        grad = _gradient(pts)
        
        # Normalize gradient magnitude to control step size?
        # Or just use fixed step.
        # Let's try a simple line search every few steps or just fixed step with decay.
        # Fixed step with decay is simpler and robust enough.
        
        new_pts = pts - step * grad
        new_pts = _project_to_sphere(new_pts)
        
        new_energy = _energy(new_pts)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
            
        if new_energy < _energy(pts):
            pts = new_pts
            # Maybe increase step slightly if we're doing well?
            # step *= 1.05
        else:
            # Reduce step
            step *= 0.5
            # If step is too small, break?
            if step < 1e-12:
                break
        
        # Decay step slowly
        step *= 0.9999
        
    # 3. Final polish: Use a tighter optimization if time remains
    # We can do a few more iterations with a very small step or use Newton's method?
    # Newton's method requires Hessian, which is expensive but N=150 is small.
    # Hessian is 450x450.
    # Let's stick to gradient descent with very small steps for polishing.
    
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # Polish phase
        step = 1e-4
        for it in range(5000):
            if time.time() > deadline:
                break
            grad = _gradient(pts)
            new_pts = pts - step * grad
            new_pts = _project_to_sphere(new_pts)
            new_energy = _energy(new_pts)
            if new_energy < _energy(pts):
                pts = new_pts
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pts = new_pts.copy()
            else:
                step *= 0.5
                if step < 1e-15:
                    break
            step *= 0.999

    return best_pts
# EVOLVE-BLOCK-END
