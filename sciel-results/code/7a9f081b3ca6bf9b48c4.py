import numpy as np
import time

N = 150

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    # Pairwise differences: (n, n, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # diff[i,j] = x_i - x_j
    dist = np.linalg.norm(diff, axis=2)  # (n, n)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j != i} diff[i,j] / dist[i,j]^2
    inv_dist_sq = 1.0 / (dist * dist)  # (n, n)
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)  # (n, 3)
    
    return energy, grad

def _project_tangent(grad, pts):
    """Project gradient onto tangent space of sphere at each point."""
    dot = np.sum(grad * pts, axis=1)  # (n,)
    return grad - dot[:, np.newaxis] * pts

def _line_search(pts, grad_t, step0, energy0):
    """Backtracking line search on the sphere."""
    step = step0
    for _ in range(50):
        # Move and re-normalize
        new_pts = pts + step * grad_t
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / norms
        
        e_new, _ = _energy_and_grad(new_pts)
        if e_new < energy0 - 1e-12 * abs(energy0):
            return new_pts, e_new, step
        step *= 0.5
    return pts, energy0, step

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize to be safe
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    energy, grad = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = energy
    
    # Gradient descent with adaptive step size
    step = 0.01
    min_step = 1e-10
    max_step = 0.1
    patience = 0
    max_patience = 200
    
    while time.time() < deadline:
        grad_t = _project_tangent(grad, pts)
        
        # Check if we've converged
        grad_norm = np.linalg.norm(grad_t)
        if grad_norm < 1e-14:
            break
        
        new_pts, e_new, step_used = _line_search(pts, grad_t, step, energy)
        
        if e_new < energy - 1e-15:
            pts = new_pts
            energy = e_new
            patience = 0
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
            
            # Increase step if we took full step
            if step_used == step:
                step = min(step * 1.1, max_step)
            else:
                step = step_used * 1.05
        else:
            step *= 0.5
            patience += 1
            if patience > max_patience:
                break
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    # Final polish with very small steps
    if time.time() < deadline - 2.0:
        step = 1e-6
        for _ in range(50):
            if time.time() > deadline - 1.0:
                break
            _, grad = _energy_and_grad(pts)
            grad_t = _project_tangent(grad, pts)
            new_pts = pts + step * grad_t
            new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
            e_new, _ = _energy_and_grad(new_pts)
            if e_new < energy:
                pts = new_pts
                energy = e_new
                if energy < best_energy:
                    best_energy = energy
                    best_pts = pts.copy()
            else:
                step *= 0.5
                if step < 1e-12:
                    break
    
    # Use best found
    result = best_pts
    # Final normalization
    result = result / np.linalg.norm(result, axis=1, keepdims=True)
    
    # Verify validity
    assert result.shape == (N, 3)
    assert np.all(np.isfinite(result))
    assert np.all(np.linalg.norm(result, axis=1) > 1e-15)
    
    return result
