"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def construct():
    # Step 1: Lattice-based Initialization
    # We model the initial configuration as a subset of a 3D lattice.
    # The lattice is defined by 3 vectors v1, v2, v3.
    # We generate a grid of points, select the 150 closest to the origin,
    # and project them onto the unit sphere.
    # We optimize the lattice vectors to minimize the energy of this projected set.
    
    # Helper to generate points from lattice vectors and project to sphere
    def generate_lattice_points(v_flat):
        v1 = v_flat[0:3]
        v2 = v_flat[3:6]
        v3 = v_flat[6:9]
        
        # Generate a 5x5x5 grid centered at 0 (indices -2 to 2)
        # Total 125 points. We need 150. So use 6x6x6 = 216 points.
        # Indices -3 to 2? Or -2 to 3? Let's use -3 to 2 for 6 steps.
        indices = np.arange(-3, 3) # -3, -2, -1, 0, 1, 2
        
        # Create meshgrid
        ix, iy, iz = np.meshgrid(indices, indices, indices, indexing='ij')
        
        # Compute points: p = ix*v1 + iy*v2 + iz*v3
        # Reshape for vectorized operation
        ix_flat = ix.flatten()
        iy_flat = iy.flatten()
        iz_flat = iz.flatten()
        
        pts = (ix_flat[:, None] * v1 + 
               iy_flat[:, None] * v2 + 
               iz_flat[:, None] * v3)
        
        # Calculate norms to find closest to origin
        norms = np.linalg.norm(pts, axis=1)
        
        # Sort by norm
        sorted_indices = np.argsort(norms)
        
        # Take the first 150
        selected_pts = pts[sorted_indices[:N]]
        
        # Project to unit sphere
        sel_norms = np.linalg.norm(selected_pts, axis=1, keepdims=True)
        # Avoid division by zero
        sel_norms = np.where(sel_norms == 0, 1.0, sel_norms)
        selected_pts = selected_pts / sel_norms
        
        return selected_pts

    # Helper to compute energy of a set of points
    def compute_energy(x):
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        mask = ~np.eye(N, dtype=bool)
        dist2_masked = np.where(mask, dist2, 1.0)
        # Add small epsilon to avoid log(0) if points are too close, though unlikely
        dist2_masked = np.maximum(dist2_masked, 1e-12)
        log_dist2 = np.log(dist2_masked)
        e_log = -0.25 * np.sum(log_dist2[mask])
        return e_log

    # Objective for lattice optimization
    def lattice_energy(v_flat):
        pts = generate_lattice_points(v_flat)
        return compute_energy(pts)

    # Initial guess for lattice vectors (cubic lattice)
    # Scale factor roughly 1.0
    v0 = np.array([1.0, 0.0, 0.0, 
                   0.0, 1.0, 0.0, 
                   0.0, 0.0, 1.0])
    
    # Optimize lattice vectors using Nelder-Mead (no gradient needed for 9 vars)
    # This step is fast and provides a symmetric, low-energy starting point.
    res_lattice = minimize(
        lattice_energy, 
        v0, 
        method='Nelder-Mead', 
        options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4}
    )
    
    # Generate initial points from optimized lattice
    x0 = generate_lattice_points(res_lattice.x)
    
    # Add tiny noise to break exact symmetries if any remain
    np.random.seed(42)
    x0 = x0 + 0.001 * np.random.randn(N, 3)
    norms = np.linalg.norm(x0, axis=1, keepdims=True)
    x0 = x0 / norms

    # Step 2: Full Coordinate Refinement
    # Use L-BFGS-B on all 450 coordinates to refine the positions.
    
    lambda_pen = 10000.0
    
    def energy_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        
        mask = ~np.eye(N, dtype=bool)
        dist2_masked = np.where(mask, dist2, 1.0)
        
        # Clip to avoid log(0)
        dist2_masked = np.maximum(dist2_masked, 1e-12)
        
        log_dist2 = np.log(dist2_masked)
        e_log = -0.25 * np.sum(log_dist2[mask])
        
        norms2 = np.sum(x ** 2, axis=1)
        pen = lambda_pen * np.sum((norms2 - 1.0) ** 2)
        
        total_energy = e_log + pen
        
        dist2_safe = np.where(mask, dist2, 1.0)
        dist2_safe = np.maximum(dist2_safe, 1e-12)
        
        grad_log = diff / dist2_safe[:, :, np.newaxis]
        grad_log = np.sum(grad_log, axis=1)
        
        grad_e = -0.5 * grad_log
        
        grad_pen = 4.0 * lambda_pen * (norms2 - 1.0)[:, np.newaxis] * x
        
        total_grad = grad_e + grad_pen
        
        return total_energy, total_grad.flatten()

    res = minimize(
        energy_and_grad, 
        x0.flatten(), 
        method='L-BFGS-B', 
        jac=True, 
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    x_opt = res.x.reshape(N, 3)
    
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    x_final = x_opt / norms
    
    return x_final

# EVOLVE-BLOCK-END
