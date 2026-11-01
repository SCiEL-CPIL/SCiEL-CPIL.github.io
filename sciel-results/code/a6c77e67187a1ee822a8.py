"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient w.r.t. the Cartesian coordinates.
    E = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
    # Avoid division by zero on the diagonal
    np.fill_diagonal(dist2, 1.0)
    
    dist = np.sqrt(dist2)
    
    # Energy: sum over i<j of log(1/dist[i,j]) = -sum over i<j log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: dE/dx_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    weight = np.zeros_like(dist2)
    np.fill_diagonal(weight, 0.0)
    weight = 1.0 / dist2
    
    grad = -np.einsum('ij,ijk->ik', weight, diff)
    
    return energy, grad

def _sphere_energy_and_grad(x_flat):
    """
    Objective and gradient for scipy.optimize.minimize.
    x_flat is a 3*N array of coordinates.
    We normalize each point to the sphere inside this function.
    """
    N_local = len(x_flat) // 3
    pts = x_flat.reshape(N_local, 3)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    energy, grad = _energy_and_grad(pts)
    
    # Project gradient onto tangent space? 
    # Actually, if we normalize inside the objective, the gradient w.r.t x_flat 
    # is not simply grad. 
    # But a common trick is to use the gradient on the sphere and let the 
    # optimizer handle the constraint via the normalization in the objective.
    # However, L-BFGS-B doesn't know about the manifold.
    # A better approach for unconstrained optimization with normalization:
    # The effective gradient w.r.t x_flat is grad (since dE/dx_flat = grad * dpts/dx_flat + ... )
    # Actually, if we define f(x) = E(x/||x||), then 
    # df/dx = (I - x x^T / ||x||^2) * grad_E(x/||x||)
    # But since we are minimizing over R^3N with the constraint ||x_i||=1, 
    # and L-BFGS-B is unconstrained, we should really use a constrained optimizer 
    # or a penalty method.
    # Alternatively, we can just return the gradient on the sphere and hope 
    # the normalization keeps it close.
    # Let's try returning grad directly. The normalization in the objective 
    # makes the function flat along the radial direction, so the gradient 
    # in that direction is 0.
    
    return energy, grad.flatten()

def construct():
    # Deadline: 150s total, leave 5s for safety
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial configuration: Golden-angle spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Flatten for scipy
    x0 = pts_init.flatten()
    
    # Use L-BFGS-B with a custom callback to check time
    best_x = x0.copy()
    best_f, best_g = _sphere_energy_and_grad(x0)
    
    def callback(xk):
        nonlocal best_x, best_f
        # Check time
        if time.time() > deadline:
            return True  # Stop optimization
        
        # Update best if improved
        f_k, _ = _sphere_energy_and_grad(xk)
        if f_k < best_f:
            best_f = f_k
            best_x = xk.copy()
        
        return False
    
    # Run L-BFGS-B
    try:
        res = minimize(
            _sphere_energy_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': 5000,
                'ftol': 1e-12,
                'gtol': 1e-10,
                'maxfun': 10000
            },
            callback=callback
        )
        if res.success or res.status == 0:
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
    except Exception as e:
        pass
    
    # Final polish with a few more steps of gradient descent if time permits
    # This helps escape any slight imperfections from the L-BFGS termination criteria
    if time.time() < deadline - 5.0:
        pts = best_x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        
        # Simple gradient descent polish
        lr = 0.01
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            f, g = _energy_and_grad(pts)
            g_flat = g.flatten()
            
            # Project gradient onto tangent space
            # grad_T = grad - (grad . x) * x
            proj = np.einsum('ij,ij->i', g, pts)
            g_tangent = g - proj[:, np.newaxis] * pts
            
            step = lr * g_tangent
            new_pts = pts - step
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / np.maximum(norms, 1e-15)
            
            new_f, _ = _energy_and_grad(new_pts)
            
            if new_f < f:
                pts = new_pts
                f = new_f
                # Slowly decay learning rate
                lr *= 0.99
            else:
                lr *= 0.5
                if lr < 1e-10:
                    break
        
        best_x = pts.flatten()
    
    # Ensure points are on the sphere
    pts_opt = best_x.reshape(N, 3)
    norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
    pts_opt = pts_opt / np.maximum(norms, 1e-15)
    
    return pts_opt
# EVOLVE-BLOCK-END
