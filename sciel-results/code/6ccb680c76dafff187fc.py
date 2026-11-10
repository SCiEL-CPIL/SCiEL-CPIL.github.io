"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|)"""
    n = pts.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return 1e18
            total += -math.log(d)
    return total

def _log_energy_vec(pts):
    """Vectorized logarithmic energy"""
    n = pts.shape[0]
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1.0)  # avoid log(0)
    # Only upper triangle
    iu = np.triu_indices(n, 1)
    d = dist[iu]
    d = np.maximum(d, 1e-15)
    return float(np.sum(-np.log(d)))

def _log_energy_grad_vec(pts):
    """Vectorized gradient of logarithmic energy on sphere (projected to tangent)"""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    # gradient[i] = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    # = sum_{j != i} -diff[i,j] / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    # Project to tangent space: grad_tangent = grad - (grad . x_i) * x_i
    dot = np.sum(grad * pts, axis=1)
    grad_tangent = grad - dot[:, np.newaxis] * pts
    return grad_tangent

def _projected_grad_desc(pts, max_iter=5000, lr_init=0.01, tol=1e-10):
    """Projected gradient descent on the sphere"""
    n = pts.shape[0]
    pts = pts.copy()
    best_pts = pts.copy()
    best_energy = _log_energy_vec(pts)
    lr = lr_init
    prev_energy = best_energy
    
    for it in range(max_iter):
        grad = _log_energy_grad_vec(pts)
        
        # Adaptive step size
        step = lr * grad
        new_pts = pts + step
        # Normalize
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / np.maximum(norms, 1e-15)
        
        new_energy = _log_energy_vec(new_pts)
        
        if new_energy < best_energy - 1e-15:
            best_pts = new_pts.copy()
            best_energy = new_energy
            pts = new_pts
            # Increase learning rate slightly if improving
            lr = min(lr * 1.05, 0.1)
        else:
            # Decrease learning rate
            lr = max(lr * 0.5, 1e-6)
            if lr <= 1e-7:
                break
        
        if abs(prev_energy - new_energy) < tol:
            break
        prev_energy = new_energy
    
    return best_pts, best_energy

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, rng):
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    best_pts = None
    best_energy = float('inf')
    
    # Multiple restarts with different initializations
    rng = np.random.RandomState(42)
    
    # Start with Fibonacci spiral
    init_configs = [_fibonacci_spiral(N)]
    
    # Add some random initializations
    for _ in range(5):
        init_configs.append(_random_points(N, rng))
    
    # Add perturbed Fibonacci
    for _ in range(5):
        base = _fibonacci_spiral(N)
        noise = rng.standard_normal((N, 3)) * 0.1
        perturbed = base + noise
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        init_configs.append(perturbed / norms)
    
    for init_idx, init_pts in enumerate(init_configs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            break
        
        # Run projected gradient descent
        result_pts, result_energy = _projected_grad_desc(
            init_pts, max_iter=3000, lr_init=0.01
        )
        
        if result_energy < best_energy:
            best_energy = result_energy
            best_pts = result_pts.copy()
        
        # Try a second pass with smaller learning rate for polishing
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            break
        
        polished_pts, polished_energy = _projected_grad_desc(
            result_pts, max_iter=2000, lr_init=0.001
        )
        
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_pts = polished_pts.copy()
    
    # Final polish with very small steps
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0 and best_pts is not None:
        final_pts, final_energy = _projected_grad_desc(
            best_pts, max_iter=1000, lr_init=0.0001
        )
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = final_pts
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    # Ensure unit norm
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
