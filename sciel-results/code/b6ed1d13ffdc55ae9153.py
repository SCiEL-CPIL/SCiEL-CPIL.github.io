import numpy as np
import time
import math

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configuration: Fibonacci spiral
    # This is a good deterministic start
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Convert to spherical coordinates
    # theta = arccos(z), phi = arctan2(y, x)
    theta = np.arccos(np.clip(x_init[:, 2], -1.0, 1.0))
    phi = np.arctan2(x_init[:, 1], x_init[:, 0])
    
    # Ensure phi in [0, 2pi)
    phi = phi % (2 * np.pi)
    
    # Avoid poles
    eps = 1e-6
    theta = np.clip(theta, eps, np.pi - eps)
    
    # Flatten for optimizer
    x0 = np.concatenate([theta, phi])
    
    def log_energy_and_grad(params):
        theta = params[:N]
        phi = params[N:]
        
        # Convert to Cartesian
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        x = sin_theta * cos_phi
        y = sin_theta * sin_phi
        z = cos_theta
        
        pts = np.column_stack([x, y, z])
        
        # Calculate distances
        # d_ij = sqrt(2 - 2 * dot(p_i, p_j))
        dots = pts @ pts.T
        np.fill_diagonal(dots, 1.0)
        
        # Clip to avoid log(0) or negative due to numerical errors
        # 1 - dot can be very small
        one_minus_dot = 1.0 - dots
        one_minus_dot = np.maximum(one_minus_dot, 1e-15)
        
        d_sq = 2.0 * one_minus_dot
        d = np.sqrt(d_sq)
        
        # Energy
        # Sum over i < j
        # Use mask or upper triangle
        iu = np.triu_indices(N, k=1)
        energy = np.sum(-np.log(d[iu]))
        
        # Gradient
        # grad_x_i = - sum_{j != i} (x_i - x_j) / d_ij^2
        # Vectorized:
        # diff[i, j] = x_i - x_j
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # d_sq matrix
        # grad_x = - sum_j diff[i,j] / d_sq[i,j]
        # Note: diagonal is 0/0, so set to 0
        inv_d_sq = np.zeros((N, N))
        inv_d_sq[iu] = 1.0 / d_sq[iu]
        # Symmetric
        inv_d_sq = inv_d_sq + inv_d_sq.T
        
        # grad_x[i] = - sum_j diff[i,j] * inv_d_sq[i,j]
        # diff[i,j] is x_i - x_j
        # sum_j (x_i - x_j) / d_sq[i,j]
        # This is a matrix multiplication: diff @ inv_d_sq?
        # diff is (N, N, 3). inv_d_sq is (N, N).
        # We want sum_j diff[i,j,k] * inv_d_sq[i,j]
        # This is einsum('ijk,ij->ik', diff, inv_d_sq)
        grad_x = -np.einsum('ijk,ij->ik', diff, inv_d_sq)
        
        # Project to spherical coordinates
        # dE/dtheta = grad_x . d(x)/dtheta
        # d(x)/dtheta = [cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta)]
        # dE/dphi = grad_x . d(x)/dphi
        # d(x)/dphi = [-sin(theta)sin(phi), sin(theta)cos(phi), 0]
        
        dtheta = np.zeros((N, 3))
        dtheta[:, 0] = cos_theta * cos_phi
        dtheta[:, 1] = cos_theta * sin_phi
        dtheta[:, 2] = -sin_theta
        
        dphi = np.zeros((N, 3))
        dphi[:, 0] = -sin_theta * sin_phi
        dphi[:, 1] = sin_theta * cos_phi
        dphi[:, 2] = 0.0
        
        grad_theta = np.einsum('ij,ij->i', grad_x, dtheta)
        grad_phi = np.einsum('ij,ij->i', grad_x, dphi)
        
        grad = np.concatenate([grad_theta, grad_phi])
        
        return energy, grad
    
    best_params = x0
    best_energy = float('inf')
    
    # Try a few restarts if time permits
    # Start with the Fibonacci spiral
    try:
        res = scipy.optimize.minimize(log_energy_and_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success and res.fun < best_energy:
            best_energy = res.fun
            best_params = res.x
    except Exception as e:
        pass
    
    # Check time
    if time.time() < deadline - 10.0:
        # Try a random restart
        rng = np.random.default_rng(42)
        theta_rand = rng.uniform(eps, np.pi - eps, N)
        phi_rand = rng.uniform(0, 2*np.pi, N)
        x0_rand = np.concatenate([theta_rand, phi_rand])
        
        try:
            res2 = scipy.optimize.minimize(log_energy_and_grad, x0_rand, method='L-BFGS-B', jac=True, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res2.success and res2.fun < best_energy:
                best_energy = res2.fun
                best_params = res2.x
        except Exception as e:
            pass
            
    # Final conversion to Cartesian
    theta_best = best_params[:N]
    phi_best = best_params[N:]
    
    sin_theta = np.sin(theta_best)
    cos_theta = np.cos(theta_best)
    cos_phi = np.cos(phi_best)
    sin_phi = np.sin(phi_best)
    
    x_final = sin_theta * cos_phi
    y_final = sin_theta * sin_phi
    z_final = cos_theta
    
    pts_final = np.column_stack([x_final, y_final, z_final])
    
    # Normalize to ensure unit length
    norms = np.linalg.norm(pts_final, axis=1, keepdims=True)
    pts_final = pts_final / norms
    
    return pts_final

# Need to import scipy
import scipy.optimize
