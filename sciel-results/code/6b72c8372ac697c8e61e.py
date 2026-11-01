import numpy as np
from math import fsum, log
from scipy.optimize import minimize

N = 150

def compute_energy_fsum(pts):
    """Energy with fsum for precision."""
    n = len(pts)
    iu = np.triu_indices(n, k=1)
    diff = pts[iu[0]] - pts[iu[1]]
    d = np.sqrt(np.sum(diff**2, axis=1))
    d = np.maximum(d, 1e-15)
    terms = np.log(1.0/d)
    return fsum(terms.tolist())

def energy_wrapper(x):
    """Wrapper for scipy minimize."""
    pts = x.reshape(N, 3)
    # Project to sphere to handle any drift during optimization
    norms = np.linalg.norm(pts, axis=1)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms[:, np.newaxis]
    return compute_energy_fsum(pts)

def construct():
    np.random.seed(42)
    
    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initialize Distance Matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(D, 1.0)
    
    # Compute initial energy
    iu = np.triu_indices(N, k=1)
    current_energy = fsum(np.log(1.0 / D[iu]).tolist())
    best_energy = current_energy
    best_pts = pts.copy()
    
    # Simulated Annealing
    T0 = 0.05
    Tf = 0.0001
    n_iter = 500000
    alpha = (Tf/T0)**(1.0/n_iter)
    T = T0
    
    # Pre-compute indices for others
    others_idx = np.arange(N)
    
    for it in range(n_iter):
        T = max(T * alpha, Tf)
        
        k = np.random.randint(N)
        
        # Generate new point via random tangent perturbation
        # This is faster than rotation matrices
        old_pt = pts[k]
        
        # Random vector in tangent space
        # v is random, project out component parallel to old_pt
        v = np.random.randn(3)
        v = v - np.dot(v, old_pt) * old_pt
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-10:
            v = np.random.randn(3)
            v = v - np.dot(v, old_pt) * old_pt
            norm_v = np.linalg.norm(v)
        v = v / norm_v
        
        # Small step size
        step = np.random.uniform(0.05, 0.2)
        new_pt = old_pt + step * v
        new_pt /= np.linalg.norm(new_pt)
        
        # Compute delta energy
        # Get indices of all points except k
        mask = np.ones(N, dtype=bool)
        mask[k] = False
        others = mask
        
        d_old = D[k, others]
        d_new = np.linalg.norm(new_pt - pts[others], axis=1)
        
        # Avoid log(0)
        d_old = np.maximum(d_old, 1e-15)
        d_new = np.maximum(d_new, 1e-15)
        
        delta_E = np.sum(np.log(1.0/d_new)) - np.sum(np.log(1.0/d_old))
        
        # Metropolis
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E/T):
            pts[k] = new_pt
            current_energy += delta_E
            
            # Update Distance Matrix
            D[k, others] = d_new
            D[others, k] = d_new
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        
        # Periodic recompute to avoid drift
        if it % 10000 == 0:
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            D = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(D, 1.0)
            current_energy = fsum(np.log(1.0 / D[iu]).tolist())
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()

    # Polish with Riemannian Gradient Descent
    pts = best_pts.copy()
    
    # Adaptive step size
    lr = 0.01
    
    for p_iter in range(2000):
        # Compute gradient for all points
        # grad[i] = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
        grads = np.zeros_like(pts)
        
        # Vectorized gradient calculation
        # diff[i, j] = x_i - x_j
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, 1.0) # Avoid division by zero
        
        # grad[i] = sum_j (x_i - x_j) / dists_sq[i, j]
        # This is sum over j of diff[i, j] / dists_sq[i, j]
        # We can do this by dividing diff by dists_sq[:, :, np.newaxis] and summing over axis=1
        # But we must exclude j=i. Since diff[i,i]=0, it contributes 0 anyway.
        inv_dists_sq = 1.0 / dists_sq
        inv_dists_sq[np.diag_indices(N)] = 0.0 # Explicitly zero out diagonal to be safe
        
        grads = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
        
        # Project gradient to tangent plane
        # g_tangent = g - (g . x) x
        dot_grads = np.sum(grads * pts, axis=1)
        grads = grads - dot_grads[:, np.newaxis] * pts
        
        # Update points
        new_pts = pts - lr * grads
        norms = np.linalg.norm(new_pts, axis=1)
        norms = np.maximum(norms, 1e-15)
        new_pts = new_pts / norms[:, np.newaxis]
        
        new_energy = compute_energy_fsum(new_pts)
        old_energy = compute_energy_fsum(pts)
        
        if new_energy < old_energy:
            pts = new_pts
            # Increase step size slightly if improving significantly?
            # Or just keep it constant/decay slowly
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.7
            if lr < 1e-8:
                break

    final_energy = compute_energy_fsum(pts)
    if final_energy < best_energy:
        best_pts = pts
    
    return best_pts
