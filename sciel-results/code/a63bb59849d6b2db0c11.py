"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150


def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. points on sphere (projected). Vectorized."""
    # pts shape (N, 3)
    # diff[i, j, k] = pts[i, k] - pts[j, k]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # dists shape (N, N)
    dists = np.linalg.norm(diff, axis=2)
    # Avoid division by zero
    np.fill_diagonal(dists, 1.0) # Diagonal is 0, we don't care about self-interaction, set to 1 to avoid inf/nan in division
    # Gradient term for pair (i, j) on point i is -diff[i, j] / dists[i, j]^2
    # Note: The total gradient for point i is sum over j != i of -diff[i, j] / dists[i, j]^2
    # Let's compute the sum over j for each i.
    # term[i, j, k] = -diff[i, j, k] / dists[i, j]^2
    # We need to sum over j (axis 1)
    
    # To avoid memory explosion for large N, N=150 is small (150*150*3 = 67500 floats), so it's fine.
    inv_d2 = 1.0 / (dists ** 2)
    # Set diagonal to 0 to exclude self
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i, k] = sum_j (-diff[i, j, k] * inv_d2[i, j])
    # diff[i, j, k] = pts[i, k] - pts[j, k]
    # So grad[i, k] = sum_j (-(pts[i, k] - pts[j, k]) * inv_d2[i, j])
    # = -pts[i, k] * sum_j(inv_d2[i, j]) + sum_j(pts[j, k] * inv_d2[i, j])
    
    # Vectorized calculation:
    # weighted_sum_pts = inv_d2 @ pts  (shape N, 3)
    # sum_inv_d2 = np.sum(inv_d2, axis=1) (shape N,)
    # grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space
    # grad[i] -= dot(grad[i], pts[i]) * pts[i]
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return grad


def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    # Compute all pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    # Avoid log(0)
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    best_pts = pts.copy()
    best_energy = _log_energy_fast(pts)
    
    # Try multiple restarts with different initializations
    restarts = [
        ("fibonacci", pts.copy()),
    ]
    
    # Add a few more diverse initializations
    # Random uniform on sphere
    rng = np.random.default_rng(42)
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        restarts.append((f"random_{seed}", pts_rand))
    
    # Refined Fibonacci with slight perturbation
    for seed in range(2):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = pts.copy()
        pts_pert += rng_local.standard_normal((N, 3)) * 0.01
        pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
        restarts.append((f"perturbed_{seed}", pts_pert))
    
    for name, init_pts in restarts:
        if time.time() > deadline - 30:
            break
        
        current_pts = init_pts.copy()
        current_energy = _log_energy_fast(current_pts)
        
        # Gradient descent with backtracking line search
        step_size = 0.01
        max_iter = 5000 # Increased since gradient is fast
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Compute gradient
            grad = _log_energy_grad(current_pts)
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm < 1e-12:
                break
            
            # Backtracking line search
            alpha = step_size
            accepted = False
            for _ in range(20):
                new_pts = current_pts - alpha * grad
                # Re-normalize to sphere
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
                
                new_energy = _log_energy_fast(new_pts)
                
                # Armijo condition
                if new_energy < current_energy - 1e-12:
                    accepted = True
                    break
                alpha *= 0.5
            
            if accepted:
                current_pts = new_pts
                current_energy = new_energy
                step_size = min(alpha * 1.05, 0.5) # Slightly increase for next time
            else:
                step_size *= 0.5
                if step_size < 1e-8:
                    break
        
        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = current_pts.copy()
        
        # Check time
        if time.time() > deadline - 15:
            break
    
    # Final polish with smaller steps if time permits
    if time.time() < deadline - 5:
        current_pts = best_pts.copy()
        current_energy = best_energy
        step_size = 0.01
        for it in range(2000):
            if time.time() > deadline - 2:
                break
            grad = _log_energy_grad(current_pts)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-14:
                break
            
            alpha = step_size
            accepted = False
            for _ in range(20):
                new_pts = current_pts - alpha * grad
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
                new_energy = _log_energy_fast(new_pts)
                if new_energy < current_energy - 1e-14:
                    accepted = True
                    break
                alpha *= 0.5
            
            if accepted:
                current_pts = new_pts
                current_energy = new_energy
                step_size = min(alpha * 1.05, 0.1)
            else:
                step_size *= 0.5
                if step_size < 1e-10:
                    break
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = current_pts
    
    # Final normalization
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
