import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

def _stereo_to_sphere(U):
    """
    Convert stereographic coordinates (u, v) to sphere points.
    Projection from North Pole (0,0,1).
    p = (2u / (1+u^2+v^2), 2v / (1+u^2+v^2), (1-u^2-v^2) / (1+u^2+v^2))
    """
    u, v = U[:, 0], U[:, 1]
    r2 = u**2 + v**2
    denom = 1.0 + r2
    px = 2.0 * u / denom
    py = 2.0 * v / denom
    pz = (1.0 - r2) / denom
    return np.column_stack([px, py, pz])

def _sphere_to_stereo(P):
    """
    Convert sphere points to stereographic coordinates.
    Inverse of _stereo_to_sphere.
    u = x / (1 - z), v = y / (1 - z)
    Note: Fails if z = 1.
    """
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    # Avoid division by zero
    denom = 1.0 - z
    # If z is close to 1, we have a problem. 
    # For robustness, if any z > 0.9999, we can flip the sphere (z -> -z) 
    # and use projection from South Pole? 
    # For now, assume we don't hit the pole exactly.
    u = x / denom
    v = y / denom
    return np.column_stack([u, v])

def _energy_and_grad_stereo(U):
    """
    Compute energy and gradient with respect to stereographic coordinates U.
    U: (N, 2)
    """
    P = _stereo_to_sphere(U)
    n = len(P)
    
    # Compute pairwise distances squared
    # diff[i, j] = P[i] - P[j]
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0) # Avoid division by zero on diagonal
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # d_ij = sqrt(dists_sq[i,j])
    # log(d_ij) = 0.5 * log(dists_sq[i,j])
    # E = -0.5 * sum_{i!=j} 0.5 * log(dists_sq[i,j]) = -0.25 * sum_{i!=j} log(dists_sq[i,j])
    # Wait, the task defines energy as sum_{i<j} log(1/d_ij).
    # sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    # sum_{i!=j} log(d_ij) = 2 * sum_{i<j} log(d_ij)
    # So E = -0.5 * sum_{i!=j} log(d_ij) = -0.25 * sum_{i!=j} log(dists_sq[i,j])
    
    log_dists_sq = np.log(dists_sq)
    np.fill_diagonal(log_dists_sq, 0.0)
    energy = -0.25 * np.sum(log_dists_sq)
    
    # Gradient in P space
    # dE/dP_i = sum_{j != i} P_j / dists_sq[i, j]
    # Let's verify.
    # E = - sum_{i<j} log(d_ij)
    # dE/dP_i = - sum_{j > i} d/dP_i log(d_ij) - sum_{j < i} d/dP_i log(d_ij)
    # d/dP_i log(d_ij) = (1/d_ij) * (1/(2 d_ij)) * 2 (P_i - P_j) = (P_i - P_j) / dists_sq
    # So dE/dP_i = - sum_{j != i} (P_i - P_j) / dists_sq[i, j] = sum_{j != i} (P_j - P_i) / dists_sq[i, j]
    # This matches the previous derivation: sum P_j / dists_sq - P_i * sum 1/dists_sq
    
    inv_dists_sq = 1.0 / dists_sq
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    # grad_P[i] = sum_j inv_dists_sq[i, j] * (P_j - P_i)
    #           = sum_j inv_dists_sq[i, j] * P_j - P_i * sum_j inv_dists_sq[i, j]
    weighted_sum_P = inv_dists_sq @ P
    sum_inv_dists_sq = np.sum(inv_dists_sq, axis=1)
    grad_P = weighted_sum_P - P * sum_inv_dists_sq[:, np.newaxis]
    
    # Jacobian of stereo map
    # p_x = 2u / (1+r2), p_y = 2v / (1+r2), p_z = (1-r2) / (1+r2)
    # J is 3x2 for each point.
    # d(p_x)/du = 2(1+r2) - 2u(2u) / (1+r2)^2 = 2(1 - v^2) / (1+r2)^2
    # d(p_x)/dv = -2u(2v) / (1+r2)^2 = -4uv / (1+r2)^2
    # d(p_y)/du = -4uv / (1+r2)^2
    # d(p_y)/dv = 2(1 - u^2) / (1+r2)^2
    # d(p_z)/du = -4u / (1+r2)^2
    # d(p_z)/dv = -4v / (1+r2)^2
    
    u, v = U[:, 0], U[:, 1]
    r2 = u**2 + v**2
    denom = (1.0 + r2)**2
    
    J_x_u = 2.0 * (1.0 - v**2) / denom
    J_x_v = -4.0 * u * v / denom
    J_y_u = -4.0 * u * v / denom
    J_y_v = 2.0 * (1.0 - u**2) / denom
    J_z_u = -4.0 * u / denom
    J_z_v = -4.0 * v / denom
    
    # grad_U = J^T grad_P
    # grad_U[i, 0] = J_x_u * grad_P[i, 0] + J_y_u * grad_P[i, 1] + J_z_u * grad_P[i, 2]
    # grad_U[i, 1] = J_x_v * grad_P[i, 0] + J_y_v * grad_P[i, 1] + J_z_v * grad_P[i, 2]
    
    grad_U = np.zeros_like(U)
    grad_U[:, 0] = J_x_u * grad_P[:, 0] + J_y_u * grad_P[:, 1] + J_z_u * grad_P[:, 2]
    grad_U[:, 1] = J_x_v * grad_P[:, 0] + J_y_v * grad_P[:, 1] + J_z_v * grad_P[:, 2]
    
    return energy, grad_U

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initializations in Sphere coords
    inits_sphere = []
    inits_sphere.append(_fibonacci_sphere(N))
    
    # Random inits
    for s in range(10):
        rng = np.random.default_rng(s)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits_sphere.append(pts)
        
    # Perturbed Fibonacci
    for s in range(10):
        rng = np.random.default_rng(s + 100)
        pts = _fibonacci_sphere(N) + rng.standard_normal((N, 3)) * 0.1
        pts = _normalize(pts)
        inits_sphere.append(pts)
        
    best_pts = None
    best_energy = float('inf')
    
    # Convert inits to Stereo
    inits_stereo = []
    for pts in inits_sphere:
        try:
            # Check for points near North Pole
            if np.any(pts[:, 2] > 0.9999):
                # Flip z for those points? Or just skip this init?
                # Let's just skip if any point is too close to North Pole to avoid huge coords
                continue
            U = _sphere_to_stereo(pts)
            inits_stereo.append(U)
        except Exception:
            continue
            
    # If no valid stereo inits, use random in stereo directly
    if not inits_stereo:
        for s in range(10):
            rng = np.random.default_rng(s)
            U = rng.standard_normal((N, 2)) * 2.0
            inits_stereo.append(U)

    # Optimize each init
    for idx, U_init in enumerate(inits_stereo):
        if time.time() > deadline - 10:
            break
            
        try:
            # L-BFGS-B
            res = minimize(_energy_and_grad_stereo, U_init, method='L-BFGS-B', jac=True, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            U_opt = res.x
            P_opt = _stereo_to_sphere(U_opt)
            P_opt = _normalize(P_opt)
            
            # Calculate energy on sphere to be safe
            # Use the fast vectorized energy
            n = len(P_opt)
            diff = P_opt[:, np.newaxis, :] - P_opt[np.newaxis, :, :]
            dists_sq = np.sum(diff * diff, axis=2)
            np.fill_diagonal(dists_sq, 1.0)
            log_dists_sq = np.log(dists_sq)
            np.fill_diagonal(log_dists_sq, 0.0)
            energy = -0.25 * np.sum(log_dists_sq)
            
            if energy < best_energy:
                best_energy = energy
                best_pts = P_opt.copy()
                best_U = U_opt.copy()
                
        except Exception as e:
            pass
            
    # Basin Hopping on the best solution
    if best_pts is not None and time.time() < deadline - 20:
        current_U = best_U.copy()
        current_energy = best_energy
        temp = 1.0
        decay = 0.95
        
        for rh in range(50):
            if time.time() > deadline - 10:
                break
            
            # Perturb U
            rng = np.random.default_rng(rh)
            noise = rng.standard_normal((N, 2)) * 0.5 * temp
            new_U = current_U + noise
            
            try:
                res = minimize(_energy_and_grad_stereo, new_U, method='L-BFGS-B', jac=True, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                U_opt = res.x
                P_opt = _stereo_to_sphere(U_opt)
                P_opt = _normalize(P_opt)
                
                n = len(P_opt)
                diff = P_opt[:, np.newaxis, :] - P_opt[np.newaxis, :, :]
                dists_sq = np.sum(diff * diff, axis=2)
                np.fill_diagonal(dists_sq, 1.0)
                log_dists_sq = np.log(dists_sq)
                np.fill_diagonal(log_dists_sq, 0.0)
                energy = -0.25 * np.sum(log_dists_sq)
                
                delta_e = energy - current_energy
                if delta_e < 0 or np.random.rand() < math.exp(-delta_e / temp):
                    current_U = U_opt
                    current_energy = energy
                    if energy < best_energy:
                        best_energy = energy
                        best_pts = P_opt.copy()
                        best_U = U_opt.copy()
                        
            except Exception:
                pass
                
            temp *= decay
            
    # Final Polish
    if best_pts is not None and time.time() < deadline - 5:
        try:
            res = minimize(_energy_and_grad_stereo, best_U, method='L-BFGS-B', jac=True, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10})
            U_opt = res.x
            P_opt = _stereo_to_sphere(U_opt)
            P_opt = _normalize(P_opt)
            
            n = len(P_opt)
            diff = P_opt[:, np.newaxis, :] - P_opt[np.newaxis, :, :]
            dists_sq = np.sum(diff * diff, axis=2)
            np.fill_diagonal(dists_sq, 1.0)
            log_dists_sq = np.log(dists_sq)
            np.fill_diagonal(log_dists_sq, 0.0)
            energy = -0.25 * np.sum(log_dists_sq)
            
            if energy < best_energy:
                best_energy = energy
                best_pts = P_opt.copy()
                
        except Exception:
            pass
            
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        
    return _normalize(best_pts)
