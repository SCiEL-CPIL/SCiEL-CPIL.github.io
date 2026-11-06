"""
Strategy: Riemannian Newton Optimization with Global Kick-And-Restart.

The current approach uses gradient descent, which is first-order and often
trapped in local minima or converges slowly near the optimum. The logarithmic
energy on the sphere has a well-defined Hessian. By using a Riemannian Newton
method, we can take steps that account for the curvature of the energy landscape
and the sphere itself, leading to much faster convergence to high-precision
local minima.

To avoid local minima, we use a multi-start strategy with "kicks" (random
perturbations) followed by aggressive Newton polishing.
"""

import numpy as np
import time
import math

N = 150
_DEADLINE = 145.0

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _log_energy(pts):
    """
    Compute sum_{i<j} log(1/|xi-xj|).
    Returns +inf if any distance < 1e-9.
    """
    n = pts.shape[0]
    G = pts @ pts.T
    iu = np.triu_indices(n, k=1)
    d_sq = 2.0 - 2.0 * G[iu]
    
    if np.any(d_sq < 1e-18):
        return 1e100
        
    d_sq = np.maximum(d_sq, 1e-15)
    return -0.5 * np.sum(np.log(d_sq))

def _log_energy_grad_hess(pts):
    """
    Compute energy, gradient, and Hessian of the log-energy on the sphere.
    
    E = sum_{i<j} log(1/|xi-xj|) = -0.5 sum_{i<j} log(d_ij^2)
    d_ij^2 = 2 - 2 G_ij
    
    Gradient in ambient R^3:
    grad_i E = sum_{j != i} (x_j - x_i) / d_ij^2
    
    Hessian in ambient R^3:
    H_ii = sum_{j != i} 2 (I - u_ij u_ij^T) / d_ij^2  where u_ij = (x_i - x_j)/d_ij
    H_ij = -2 u_ij u_ij^T / d_ij^2  for i != j
    
    Note: We need the Riemannian Hessian, which is the projection of the ambient
    Hessian onto the tangent space.
    P_i = I - x_i x_i^T
    H_riem = P_i H_amb P_i (conceptually, but we operate in tangent vectors).
    
    For the Newton step, we solve:
    H_riem v = -grad_riem
    where v is in the tangent space.
    """
    n = pts.shape[0]
    G = pts @ pts.T
    
    # d_sq[i,j]
    d_sq = 2.0 - 2.0 * G
    np.fill_diagonal(d_sq, 1.0)
    
    # Weights W[i,j] = 1/d_sq[i,j]
    W = 1.0 / d_sq
    np.fill_diagonal(W, 0.0)
    
    # Gradient
    w_sum = W.sum(axis=1)
    weighted_pts = W @ pts
    grad_amb = weighted_pts - pts * w_sum[:, np.newaxis]
    
    # Project gradient to tangent space
    grad_dot_x = np.einsum('ij,ij->i', grad_amb, pts)
    grad_tan = grad_amb - pts * grad_dot_x[:, np.newaxis]
    
    # Hessian
    # H_amb is a block matrix.
    # Diagonal blocks H_ii = sum_{j != i} 2 (I - u_ij u_ij^T) / d_ij^2
    # Off-diagonal blocks H_ij = -2 u_ij u_ij^T / d_ij^2
    
    # We will build the Riemannian Hessian acting on tangent vectors.
    # Instead of building the full 3N x 3N matrix, we can compute its action.
    # But for N=150, 450x450 is manageable. Let's build it explicitly for clarity and use np.linalg.solve.
    
    H_riem = np.zeros((3*N, 3*N))
    
    # Precompute u_ij and weights
    # u_ij = (x_i - x_j) / d_ij
    # d_ij = sqrt(d_sq[i,j])
    
    # To save time, we only compute upper triangle and symmetrize.
    iu = np.triu_indices(n, k=1)
    i_idx, j_idx = iu
    d_vals = np.sqrt(d_sq[iu])
    u_vecs = (pts[i_idx] - pts[j_idx]) / d_vals[:, np.newaxis]
    w_vals = 2.0 / d_sq[iu]
    
    # Off-diagonal blocks H_ij = -w * u u^T
    # H_ii = sum_{j != i} w * (I - u u^T)
    
    # Initialize H_riem
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        w = w_vals[k]
        u = u_vecs[k]
        uuT = np.outer(u, u)
        
        # H_ij block (and H_ji)
        block_off = -w * uuT
        
        # H_ii contribution
        block_diag_i = w * (np.eye(3) - uuT)
        # H_jj contribution
        block_diag_j = w * (np.eye(3) - uuT)
        
        # Add to H_amb first, then project?
        # Actually, H_riem = P H_amb P.
        # Since grad_tan is in tangent space, we can just project the Hessian action.
        # But building the full projected Hessian is easier to solve.
        
        H_riem[3*i:3*i+3, 3*i:3*i+3] += block_diag_i
        H_riem[3*j:3*j+3, 3*j:3*j+3] += block_diag_j
        H_riem[3*i:3*i+3, 3*j:3*j+3] += block_off
        H_riem[3*j:3*j+3, 3*i:3*i+3] += block_off.T # uuT is symmetric
        
    # Project H_riem to tangent space: P_i H_riem P_j
    # P_i = I - x_i x_i^T
    # This is expensive to do explicitly for all blocks.
    # Alternative: Solve (H_amb + lambda I) v = -grad_amb, then project v?
    # No, Riemannian Newton is: find v in T_x M such that Hess E (v) = -grad E.
    # Since grad E is already projected, we can restrict H to tangent space.
    
    # Let's construct the tangent basis for each point to map to local coordinates?
    # Or just project the Hessian matrix.
    # P_i = I - x_i x_i^T
    # H_riem_block(i,j) = P_i H_amb_block(i,j) P_j
    
    # This requires 3x3 projections for each pair. N=150 -> 11k pairs. 3x3 mult is cheap.
    
    H_tan = np.zeros((3*N, 3*N))
    for i in range(n):
        Pi = np.eye(3) - np.outer(pts[i], pts[i])
        for j in range(i, n):
            if i == j:
                Hij = H_riem[3*i:3*i+3, 3*j:3*j+3]
                H_tan_block = Pi @ Hij @ Pi.T
                H_tan[3*i:3*i+3, 3*i:3*i+3] = H_tan_block
            else:
                Hij = H_riem[3*i:3*i+3, 3*j:3*j+3]
                Pj = np.eye(3) - np.outer(pts[j], pts[j])
                H_tan_block = Pi @ Hij @ Pj.T
                H_tan[3*i:3*i+3, 3*j:3*j+3] = H_tan_block
                H_tan[3*j:3*j+3, 3*i:3*i+3] = H_tan_block.T

    # Add small regularization for stability
    # The Hessian of log energy is not positive definite everywhere, but near minima it should be.
    # We add a small ridge to ensure invertibility.
    H_tan += 1e-6 * np.eye(3*N)
    
    # Solve H_tan v = -grad_tan
    # Flatten grad_tan
    grad_flat = grad_tan.flatten()
    
    try:
        v_flat = np.linalg.solve(H_tan, -grad_flat)
    except np.linalg.LinAlgError:
        # Fallback to gradient step if Hessian is singular
        v_flat = -grad_flat
        
    v = v_flat.reshape(n, 3)
    
    # Project v to tangent space again (numerical safety)
    v_dot_x = np.einsum('ij,ij->i', v, pts)
    v_tan = v - pts * v_dot_x[:, np.newaxis]
    
    return _log_energy(pts), grad_tan, v_tan

def _newton_step(pts, step_scale=1.0):
    """
    Perform a Riemannian Newton step with line search.
    """
    energy, grad, v_newton = _log_energy_grad_hess(pts)
    
    # Normalize step direction to control magnitude
    v_norm = np.linalg.norm(v_newton)
    if v_norm < 1e-12:
        return pts, energy
        
    v_dir = v_newton / v_norm
    
    # Line search
    best_pts = pts
    best_energy = energy
    
    # Start with a moderate step size
    step_sizes = [step_scale * v_norm, 0.5 * step_scale * v_norm, 0.25 * step_scale * v_norm, 
                  0.1 * step_scale * v_norm, 0.05 * step_scale * v_norm, 0.01 * step_scale * v_norm]
    
    for step in step_sizes:
        # Move in tangent direction
        new_pts = pts + step * v_dir
        new_pts = _normalize(new_pts)
        
        new_energy = _log_energy(new_pts)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts
            
    return best_pts, best_energy

def _fibonacci_sphere(n, offset=0.0):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n):
    vecs = np.random.normal(size=(n, 3))
    return _normalize(vecs)

def construct():
    start_time = time.time()
    
    best_pts = None
    best_energy = float('inf')
    
    # Multi-start
    starts = []
    starts.append(_fibonacci_sphere(N))
    for offset in [0.1, 0.5, 1.0, 2.0, 3.0]:
        starts.append(_fibonacci_sphere(N, offset))
    for _ in range(5):
        starts.append(_random_sphere(N))
    
    step_scale = 0.1
    
    for init_pts in starts:
        if time.time() > _DEADLINE:
            break
            
        current_pts = init_pts.copy()
        current_energy = _log_energy(current_pts)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = current_pts.copy()
            
        # Newton Optimization Loop
        # We want to converge quickly.
        for i in range(100):
            if time.time() > _DEADLINE:
                break
                
            prev_energy = current_energy
            current_pts, current_energy = _newton_step(current_pts, step_scale)
            
            # Check for convergence
            if abs(prev_energy - current_energy) < 1e-12:
                break
                
            # Adaptive step scale: if energy didn't improve much, reduce scale
            # If it improved a lot, maybe keep it or increase slightly?
            # For Newton, step size is often determined by the Hessian.
            # The line search handles the magnitude.
            
            # If stuck, kick
            if abs(prev_energy - current_energy) < 1e-6 and i > 10:
                # Random kick
                kick = np.random.normal(size=(N, 3))
                kick = _normalize(kick)
                # Blend
                current_pts = _normalize(current_pts * 0.9 + kick * 0.1)
                current_energy = _log_energy(current_pts)
                
                # Reset step scale
                step_scale = 0.5
                
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = current_pts.copy()
            
        # If we have time, try a "kick and polish" on the best found so far
        if best_pts is not None and time.time() < _DEADLINE - 10:
             kick_pts = _normalize(best_pts * 0.95 + np.random.normal(size=(N, 3)) * 0.05)
             kick_energy = _log_energy(kick_pts)
             
             # Polish the kick
             for j in range(50):
                 if time.time() > _DEADLINE:
                     break
                 kick_pts, kick_energy = _newton_step(kick_pts, 0.1)
                 
             if kick_energy < best_energy:
                 best_energy = kick_energy
                 best_pts = kick_pts.copy()

    # Final verification
    if best_pts is None:
        return _fibonacci_sphere(N)
        
    # Check validity
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    if np.any(d < 1e-9):
        return _fibonacci_sphere(N)
        
    return best_pts
