import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # 1. Initial Configuration: 5x5x4 grid
    # This gives exactly 100 points.
    # x, y have 5 points, z has 4 points.
    # Spacing: 1/5 = 0.2 for x,y. 1/4 = 0.25 for z.
    # Centered at origin.
    
    def make_grid(nx, ny, nz):
        xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
        ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        return np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Base grid
    c0 = make_grid(5, 5, 4)
    
    # We will try a few variations: 
    # 1. Pure grid
    # 2. Jittered grid
    # 3. Maybe 4x5x5 grid? (Same dimensions, just swapped axes)
    
    candidates = [c0.copy()]
    
    # Add a jittered version
    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.01, c0.shape)
    c1 = c0 + jitter
    # Ensure bounds are respected (clip)
    c1 = np.clip(c1, -0.5, 0.5)
    candidates.append(c1)
    
    # Add a 4x5x5 grid
    c2 = make_grid(4, 5, 5)
    candidates.append(c2)
    
    # Add a jittered 4x5x5
    jitter2 = rng.normal(0, 0.01, c2.shape)
    c3 = c2 + jitter2
    c3 = np.clip(c3, -0.5, 0.5)
    candidates.append(c3)
    
    best_radius = 0.0
    best_c = None
    
    def compute_radius(c):
        # c is (N, 3)
        # Min distance between pairs
        # Use cdist for efficiency? Or manual.
        # N=100, 4950 pairs. Manual is fine.
        dists = np.zeros((N, N))
        for i in range(N):
            diff = c - c[i]
            dists[i] = np.sqrt(np.sum(diff**2, axis=1))
        # Zero out diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Min distance to walls
        # |c_ik| <= 0.5
        # Distance to wall is 0.5 - |c_ik|
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def objective(c_flat):
        c = c_flat.reshape(N, 3)
        # Smooth approximation of minimizing 1/r
        # We want to maximize r.
        # Minimize sum of 1/d^p
        p = 10
        eps = 1e-6
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        dists_sq = dists_sq[mask]
        
        # Wall distances
        wall_dists_sq = (0.5 - np.abs(c))**2
        wall_dists_sq = wall_dists_sq.ravel()
        
        # Combine
        # If any distance is very small, objective explodes.
        # We add eps to avoid inf.
        obj_pairs = np.sum(1.0 / (dists_sq + eps)**(p/2.0))
        obj_walls = np.sum(1.0 / (wall_dists_sq + eps)**(p/2.0))
        
        return obj_pairs + obj_walls

    def gradient(c_flat):
        c = c_flat.reshape(N, 3)
        p = 10
        eps = 1e-6
        grad = np.zeros_like(c)
        
        # Gradient of pairwise term:
        # f_ij = (d_ij^2 + eps)^(-p/2)
        # df/dc_i = -p/2 * (d_ij^2 + eps)^(-p/2 - 1) * 2 * (c_i - c_j)
        # Sum over j
        
        # Vectorized gradient for pairs
        # This is O(N^2).
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists_sq = np.maximum(dists_sq, eps) # Avoid 0
        factor = -p * (dists_sq ** (-p/2.0 - 1)) # -p/2 * 2
        # Multiply by diff (c_i - c_j)
        grad_pairs = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        # Gradient of wall term:
        # f_ik = (0.5 - |c_ik|)^2 + eps)^(-p/2)
        # Let u_ik = 0.5 - |c_ik|
        # df/du = -p/2 * (u^2+eps)^(-p/2-1) * 2u = -p * u * (u^2+eps)^(-p/2-1)
        # du/dc_ik = -sign(c_ik)
        # df/dc_ik = p * sign(c_ik) * u * (u^2+eps)^(-p/2-1)
        
        abs_c = np.abs(c)
        u = 0.5 - abs_c
        u = np.maximum(u, 1e-6) # Avoid negative/zero issues in power
        wall_factor = p * u * (u**2 + eps)**(-p/2.0 - 1)
        grad_walls = wall_factor * np.sign(c)
        
        return (grad_pairs + grad_walls).ravel()

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Evaluate initial candidates
    for c in candidates:
        r = compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_c = c.copy()
            
    # Optimize each candidate
    for i, c in enumerate(candidates):
        if time.time() > deadline:
            break
            
        c_flat = c.ravel()
        try:
            res = minimize(objective, c_flat, jac=gradient, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            c_opt = res.x.reshape(N, 3)
            # Clip to ensure validity
            c_opt = np.clip(c_opt, -0.5, 0.5)
            r_opt = compute_radius(c_opt)
            
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
                
        except Exception as e:
            pass

    # If time remains, do a final polish on the best candidate with higher precision
    if best_c is not None and time.time() < deadline - 10:
        c_flat = best_c.ravel()
        try:
            # Increase p for sharper minimum approximation
            p_refine = 20
            def objective_refine(c_flat):
                c = c_flat.reshape(N, 3)
                eps = 1e-8
                diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
                dists_sq = np.sum(diff**2, axis=2)
                mask = ~np.eye(N, dtype=bool)
                dists_sq = dists_sq[mask]
                wall_dists_sq = (0.5 - np.abs(c))**2
                wall_dists_sq = wall_dists_sq.ravel()
                
                obj_pairs = np.sum(1.0 / (dists_sq + eps)**(p_refine/2.0))
                obj_walls = np.sum(1.0 / (wall_dists_sq + eps)**(p_refine/2.0))
                return obj_pairs + obj_walls

            def gradient_refine(c_flat):
                c = c_flat.reshape(N, 3)
                eps = 1e-8
                diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
                dists_sq = np.sum(diff**2, axis=2)
                dists_sq = np.maximum(dists_sq, eps)
                factor = -p_refine * (dists_sq ** (-p_refine/2.0 - 1))
                grad_pairs = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
                
                abs_c = np.abs(c)
                u = 0.5 - abs_c
                u = np.maximum(u, 1e-6)
                wall_factor = p_refine * u * (u**2 + eps)**(-p_refine/2.0 - 1)
                grad_walls = wall_factor * np.sign(c)
                
                return (grad_pairs + grad_walls).ravel()

            res = minimize(objective_refine, c_flat, jac=gradient_refine, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50})
            c_opt = res.x.reshape(N, 3)
            c_opt = np.clip(c_opt, -0.5, 0.5)
            r_opt = compute_radius(c_opt)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
        except Exception as e:
            pass

    if best_c is None:
        return c0
        
    return best_c

# EVOLVE-BLOCK-END
