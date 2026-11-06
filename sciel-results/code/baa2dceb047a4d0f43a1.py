"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_energy = float('inf')
    best_points = None
    
    def energy_and_grad(x_flat):
        """
        Computes the logarithmic energy and Euclidean gradient for L-BFGS-B.
        We work in R^3N and project the gradient to the tangent space.
        """
        x = x_flat.reshape(N, 3)
        # Compute pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)  # Avoid div by 0
        
        # Energy: -sum_{i<j} log(|x_i - x_j|)
        # sum_{i!=j} log(d_ij) = 2 * sum_{i<j} log(d_ij)
        dist = np.sqrt(dist_sq)
        mask = ~np.eye(N, dtype=bool)
        log_d = np.log(dist)
        log_d[~mask] = 0.0
        energy = -0.5 * np.sum(log_d)
        
        # Gradient: grad_E_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
        inv_dist_sq = 1.0 / dist_sq
        F = diff * inv_dist_sq[:, :, np.newaxis]
        grad_E = -np.sum(F, axis=1)
        
        # Project to tangent space: grad_tangent = grad_E - (grad_E . x_i) * x_i
        dot = np.sum(grad_E * x, axis=1, keepdims=True)
        grad_tangent = grad_E - dot * x
        
        return energy, grad_tangent.flatten()

    def energy_only(x_flat):
        """Energy only for line search."""
        x = x_flat.reshape(N, 3)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        mask = ~np.eye(N, dtype=bool)
        log_d = np.log(dist)
        log_d[~mask] = 0.0
        return -0.5 * np.sum(log_d)

    def lbfgs_optimize(x0, max_time=30.0):
        """Run L-BFGS-B with time limit."""
        x0_flat = x0.flatten()
        result = minimize(
            energy_and_grad,
            x0_flat,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
        )
        x_opt = result.x.reshape(N, 3)
        # Normalize to ensure unit sphere
        x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
        E_opt, _ = energy_and_grad(x_opt.flatten())
        return x_opt, E_opt

    def riemannian_polish(x, max_iters=500):
        """Final Riemannian gradient descent polish."""
        x = x.copy()
        E, g = energy_and_grad(x.flatten())
        g = g.reshape(N, 3)
        
        for k in range(max_iters):
            if time.time() > deadline:
                break
            
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-12:
                break
            
            # Backtracking line search
            eta = 0.01
            improved = False
            for _ in range(30):
                x_new = x - eta * g
                x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)
                E_new, _ = energy_and_grad(x_new.flatten())
                
                if E_new < E - 1e-4 * eta * g_norm:
                    improved = True
                    break
                eta *= 0.5
            
            if not improved:
                break
            
            x = x_new
            E = E_new
            _, g_flat = energy_and_grad(x.flatten())
            g = g_flat.reshape(N, 3)
        
        return x, E

    # Generate multiple starting configurations
    starts = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    starts.append(x_fib)
    
    # 2-5. Perturbed Fibonacci with different noise levels and seeds
    for seed, noise_scale in [(42, 0.05), (123, 0.1), (456, 0.15), (789, 0.2)]:
        if time.time() > deadline - 10:
            break
        np.random.seed(seed)
        noise = np.random.normal(0, noise_scale, (N, 3))
        x_pert = x_fib + noise
        x_pert = x_pert / np.linalg.norm(x_pert, axis=1, keepdims=True)
        starts.append(x_pert)
    
    # 6-8. Random points
    for seed in [111, 222, 333]:
        if time.time() > deadline - 10:
            break
        np.random.seed(seed)
        x_rand = np.random.normal(0, 1, (N, 3))
        x_rand = x_rand / np.linalg.norm(x_rand, axis=1, keepdims=True)
        starts.append(x_rand)
    
    # 9-10. Spherical spiral variations
    for seed in [444, 555]:
        if time.time() > deadline - 10:
            break
        np.random.seed(seed)
        # Generate points on sphere using random directions
        x_spiral = np.random.normal(0, 1, (N, 3))
        x_spiral = x_spiral / np.linalg.norm(x_spiral, axis=1, keepdims=True)
        starts.append(x_spiral)
    
    # Optimize each start
    for idx, x_start in enumerate(starts):
        if time.time() > deadline - 15:
            break
        
        # L-BFGS-B optimization
        x_opt, E_opt = lbfgs_optimize(x_start)
        
        # Riemannian polish
        x_polished, E_polished = riemannian_polish(x_opt)
        
        if E_polished < best_energy:
            best_energy = E_polished
            best_points = x_polished.copy()
    
    # Final polish on best result
    if time.time() < deadline - 5 and best_points is not None:
        x_final, E_final = riemannian_polish(best_points, max_iters=200)
        if E_final < best_energy:
            best_energy = E_final
            best_points = x_final
    
    return best_points

# EVOLVE-BLOCK-END
