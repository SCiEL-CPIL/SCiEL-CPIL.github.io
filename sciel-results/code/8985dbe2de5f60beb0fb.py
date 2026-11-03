"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    # Energy function and gradient on the sphere (using retraction approach)
    # We optimize in R^3 but project back to sphere after each gradient step
    # This is more efficient than penalty methods
    
    def energy_and_grad(x):
        """Compute energy and gradient, with x on sphere (approximately)."""
        # Ensure x is on sphere
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        
        # Pairwise differences
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)  # Avoid zero division
        
        # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
        # = -0.5 * sum_{i<j} log(dist_sq)
        triu_idx = np.triu_indices(N, 1)
        energy = -0.5 * np.sum(np.log(dist_sq[triu_idx]))
        
        # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / dist_sq_ij
        # Note: this is the gradient in R^3 space
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # grad[i] = sum_j diff[i,j] * inv_dist_sq[i,j]
        grad = np.einsum('ijk,ij->ik', diff, inv_dist_sq)
        
        return energy, grad
    
    def project_to_sphere(x):
        """Project points back to unit sphere."""
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        return x / norms
    
    def compute_energy(x):
        """Compute just the energy for evaluation."""
        x = project_to_sphere(x)
        e, _ = energy_and_grad(x)
        return e
    
    def compute_gradient(x):
        """Compute just the gradient for optimization."""
        x = project_to_sphere(x)
        _, g = energy_and_grad(x)
        # Project gradient to tangent space of sphere
        g_dot_x = np.sum(g * x, axis=1, keepdims=True)
        g_tangent = g - g_dot_x * x
        return g_tangent.flatten()
    
    # Generate multiple initial configurations and optimize each
    best_energy = np.inf
    best_x = None
    
    # Different initialization strategies
    init_configs = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    init_configs.append(fib)
    
    # 2. Random points on sphere with various seeds
    for seed in [1, 2, 3, 4, 5, 42, 123, 456]:
        rng_local = np.random.default_rng(seed)
        pts = rng_local.normal(size=(N, 3))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        init_configs.append(pts)
    
    # 3. Fibonacci with various jitter scales
    for jitter_scale in [0.01, 0.02, 0.05, 0.1, 0.2]:
        rng_local = np.random.default_rng(100 + int(jitter_scale * 1000))
        jittered = fib + rng_local.normal(0, jitter_scale, fib.shape)
        jittered = jittered / np.linalg.norm(jittered, axis=1, keepdims=True)
        init_configs.append(jittered)
    
    # 4. Spherical grid with perturbation
    n_theta = 10
    n_phi = 10
    thetas = np.linspace(0, np.pi, n_theta + 2)[1:-1]
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    grid_pts = []
    for th in thetas:
        for ph in phis:
            x = np.sin(th) * np.cos(ph)
            y = np.sin(th) * np.sin(ph)
            z_val = np.cos(th)
            grid_pts.append([x, y, z_val])
    grid_pts = np.array(grid_pts[:N])
    if len(grid_pts) < N:
        # Pad with random points if needed
        rng_local = np.random.default_rng(999)
        extra = rng_local.normal(size=(N - len(grid_pts), 3))
        extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
        grid_pts = np.vstack([grid_pts, extra])
    init_configs.append(grid_pts)
    
    # Optimize each configuration with multi-stage approach
    for init_idx, x0 in enumerate(init_configs):
        x0 = project_to_sphere(x0)
        
        # Stage 1: Coarse optimization with larger tolerance
        try:
            res1 = minimize(
                lambda xf: compute_energy(xf.reshape(N, 3)),
                x0.flatten(),
                jac=lambda xf: compute_gradient(xf.reshape(N, 3)),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            x_stage1 = project_to_sphere(res1.x.reshape(N, 3))
        except Exception:
            x_stage1 = x0
        
        # Stage 2: Medium optimization
        try:
            res2 = minimize(
                lambda xf: compute_energy(xf.reshape(N, 3)),
                x_stage1.flatten(),
                jac=lambda xf: compute_gradient(xf.reshape(N, 3)),
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-8}
            )
            x_stage2 = project_to_sphere(res2.x.reshape(N, 3))
        except Exception:
            x_stage2 = x_stage1
        
        # Stage 3: Fine optimization
        try:
            res3 = minimize(
                lambda xf: compute_energy(xf.reshape(N, 3)),
                x_stage2.flatten(),
                jac=lambda xf: compute_gradient(xf.reshape(N, 3)),
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            x_opt = project_to_sphere(res3.x.reshape(N, 3))
        except Exception:
            x_opt = x_stage2
        
        e_final = compute_energy(x_opt)
        
        if e_final < best_energy:
            best_energy = e_final
            best_x = x_opt.copy()
    
    # Final ultra-fine refinement from the best solution
    if best_x is not None:
        # Run multiple refinement passes with increasingly tight tolerances
        for tol in [1e-12, 1e-14, 1e-16]:
            try:
                res_final = minimize(
                    lambda xf: compute_energy(xf.reshape(N, 3)),
                    best_x.flatten(),
                    jac=lambda xf: compute_gradient(xf.reshape(N, 3)),
                    method='L-BFGS-B',
                    options={'maxiter': 10000, 'ftol': tol, 'gtol': tol * 1e-2}
                )
                best_x = project_to_sphere(res_final.x.reshape(N, 3))
            except Exception:
                break
    
    # Ensure output is valid
    best_x = project_to_sphere(best_x)
    
    # Verify no points are too close
    diff = best_x[:, np.newaxis, :] - best_x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 1e-9:
        # Add small perturbation if points are too close
        rng = np.random.default_rng(123)
        best_x = best_x + rng.normal(0, 1e-6, best_x.shape)
        best_x = project_to_sphere(best_x)
    
    return best_x
# EVOLVE-BLOCK-END
