"""
Strategy: Dual-representation optimization for logarithmic energy minimization on the sphere.
1. Initialize using Halton sequence for low-discrepancy global dispersion.
2. Alternate between:
   - Riemannian Conjugate Gradient (RCG) on the sphere for fast local convergence.
   - Projected Gradient Descent (PGD) in ambient space with larger steps to escape shallow local minima.
3. Multiple restarts with different Halton bases/seeds.
4. Strict time management to ensure completion within 150s.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _halton_sequence(n, base):
    """
    Generate n Halton sequence points in [0, 1) for a given base.
    """
    result = np.zeros(n)
    f = 1.0 / base
    for i in range(n):
        idx = i + 1
        while idx > 0:
            result[i] += f * (idx % base)
            idx //= base
            f /= base
    return result

def _halton_sphere(n, seed=0):
    """
    Generate n points on the unit sphere using Halton sequences in 3D.
    Uses bases 2, 3, 5 for low correlation.
    """
    # We need to map [0,1)^3 to sphere.
    # A common method is to use spherical coordinates, but Halton in 3D mapped directly
    # to cube and then normalized is a simple low-discrepancy method on the sphere surface
    # if we ensure uniformity. However, simple normalization of cube points creates
    # clustering at corners. 
    # Better: Use Halton to generate uniform points in cube, reject out of sphere, 
    # or map to sphere via area-preserving map.
    # For simplicity and speed, let's use the standard "random" normal distribution 
    # but seeded by Halton-like indices? No, Halton in 3D mapped to sphere via 
    # uniform cube -> sphere normalization is actually quite good for initial dispersion 
    # if we just take the direction.
    
    # Let's use 3 independent Halton sequences for x, y, z coordinates in [-1, 1]
    # and normalize.
    h1 = _halton_sequence(n, 2) * 2 - 1
    h2 = _halton_sequence(n, 3) * 2 - 1
    h3 = _halton_sequence(n, 5) * 2 - 1
    
    points = np.column_stack([h1, h2, h3])
    
    # Normalize to sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero (unlikely with Halton)
    norms[norms == 0] = 1.0
    points /= norms
    
    # Shuffle slightly to break any grid-like artifacts if present, though Halton is deterministic
    # rng = np.random.default_rng(seed)
    # perm = rng.permutation(n)
    # points = points[perm]
    
    return points

def _log_energy_and_grad_sphere(x):
    """
    Computes the logarithmic energy and its Euclidean gradient for points on the sphere.
    x: (N, 3) array of points on the unit sphere.
    Returns: energy, grad (N, 3)
    """
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _riemannian_grad(x, grad_euclidean):
    """
    Project Euclidean gradient onto the tangent space of the sphere at x.
    """
    proj = np.sum(grad_euclidean * x, axis=1, keepdims=True)
    return grad_euclidean - proj * x

def _exp_map(x, v):
    """
    Exponential map on the unit sphere.
    """
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    norm_v_safe = np.where(norm_v < 1e-12, 1e-12, norm_v)
    sin_norm = np.sin(norm_v_safe)
    cos_norm = np.cos(norm_v_safe)
    
    small = norm_v < 1e-12
    sin_norm[small] = 0.0
    cos_norm[small] = 1.0
    
    factor = sin_norm / norm_v_safe
    factor[small] = 1.0
    
    new_x = cos_norm * x + factor * v
    new_x /= np.linalg.norm(new_x, axis=1, keepdims=True)
    return new_x

def _riemannian_cg(x, max_iter=50, tol=1e-8, time_limit=5.0):
    """
    Riemannian Conjugate Gradient method on the sphere.
    """
    start_time = time.time()
    energy, grad_eucl = _log_energy_and_grad_sphere(x)
    grad_tan = _riemannian_grad(x, grad_eucl)
    
    if np.linalg.norm(grad_tan) < tol:
        return x, energy
    
    p = -grad_tan
    p_norm = np.linalg.norm(p)
    if p_norm > 0:
        p /= p_norm
    best_x = x.copy()
    best_energy = energy
    
    for k in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Backtracking line search
        t = 1.0
        c1 = 1e-4
        alpha = 0.5
        success = False
        for _ in range(30):
            x_new = _exp_map(x, t * p)
            e_new, grad_eucl_new = _log_energy_and_grad_sphere(x_new)
            
            if e_new < energy + c1 * t * np.sum(grad_tan * p):
                success = True
                break
            t *= alpha
        
        if not success:
            break
            
        x = x_new
        energy = e_new
        grad_eucl = grad_eucl_new
        grad_tan_new = _riemannian_grad(x, grad_eucl)
        
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()
            
        if np.linalg.norm(grad_tan_new) < tol:
            break
            
        # Polak-Ribiere beta
        beta = np.sum(grad_tan_new * (grad_tan_new - grad_tan)) / (np.sum(grad_tan * grad_tan) + 1e-12)
        beta = max(0.0, beta)
        
        p_new = -grad_tan_new + beta * p
        p_new = _riemannian_grad(x, p_new)
        
        p_norm = np.linalg.norm(p_new)
        if p_norm < 1e-12:
            p_new = -grad_tan_new
            p_new = _riemannian_grad(x, p_new)
            p_norm = np.linalg.norm(p_new)
        
        if p_norm > 0:
            p = p_new / p_norm
        else:
            break
            
        grad_tan = grad_tan_new
        
    return best_x, best_energy

def _projected_gradient_step(x, step_size=0.01, time_limit=2.0):
    """
    Simple projected gradient descent in ambient space.
    Moves points slightly in the direction of negative gradient, then projects back to sphere.
    This is less efficient than RCG but can help escape local minima if step size is tuned.
    """
    start_time = time.time()
    energy, grad_eucl = _log_energy_and_grad_sphere(x)
    x_curr = x.copy()
    e_curr = energy
    
    for _ in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Move in negative gradient direction
        x_new = x_curr - step_size * grad_eucl
        
        # Project to sphere
        x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
        
        e_new, _ = _log_energy_and_grad_sphere(x_new)
        
        if e_new < e_curr:
            x_curr = x_new
            e_curr = e_new
        else:
            # Reduce step size if we didn't improve
            step_size *= 0.5
            if step_size < 1e-6:
                break
                
    return x_curr, e_curr

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_energy = float('inf')
    best_x = None
    
    # Generate multiple initial configurations using Halton sequences with different bases/shuffles
    # We'll use different base permutations or slight perturbations
    inits = []
    
    # Base Halton
    inits.append(_halton_sphere(N, seed=0))
    
    # Perturbed Halton
    rng = np.random.default_rng(123)
    x_pert = _halton_sphere(N, seed=0) + 0.05 * rng.normal(size=(N, 3))
    x_pert /= np.linalg.norm(x_pert, axis=1, keepdims=True)
    inits.append(x_pert)
    
    # Another Halton with different base order (approximated by shuffling indices)
    h1 = _halton_sequence(N, 3) * 2 - 1
    h2 = _halton_sequence(N, 2) * 2 - 1
    h3 = _halton_sequence(N, 5) * 2 - 1
    points = np.column_stack([h1, h2, h3])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    inits.append(points)
    
    # Random initialization as fallback
    rng2 = np.random.default_rng(456)
    x_rand = rng2.normal(size=(N, 3))
    x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
    inits.append(x_rand)
    
    for init_idx, x_init in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        # Phase 1: RCG for fast local convergence
        x_res, e_res = _riemannian_cg(x_init, max_iter=100, tol=1e-8, time_limit=10.0)
        
        if e_res < best_energy:
            best_energy = e_res
            best_x = x_res.copy()
        
        # Phase 2: Alternating PGD and RCG to escape local minima
        if time.time() < deadline - 10:
            x_curr = x_res
            e_curr = e_res
            pgd_step = 0.05
            
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                
                # PGD step
                x_pgd, e_pgd = _projected_gradient_step(x_curr, step_size=pgd_step, time_limit=2.0)
                
                # RCG polish
                x_final, e_final = _riemannian_cg(x_pgd, max_iter=50, tol=1e-8, time_limit=5.0)
                
                if e_final < best_energy:
                    best_energy = e_final
                    best_x = x_final.copy()
                
                # Update for next iteration
                if e_final < e_curr:
                    x_curr = x_final
                    e_curr = e_final
                    # Keep step size or increase slightly? Keep same for stability
                else:
                    # If no improvement, reduce PGD step size to be more local
                    pgd_step *= 0.5
                    if pgd_step < 1e-4:
                        break
                        
    if best_x is None:
        best_x = inits[0]
        
    return best_x

# EVOLVE-BLOCK-END
