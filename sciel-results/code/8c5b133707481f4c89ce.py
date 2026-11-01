import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _energy_and_grad(theta_phi):
    """
    theta_phi: shape (N, 2), theta in [0, pi], phi in [0, 2pi]
    Returns energy and gradient.
    """
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]
    
    # Compute coordinates
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = cos_t
    
    pts = np.column_stack((x, y, z))
    
    # Compute pairwise distances squared
    # diff[i, j] = pts[i] - pts[j]
    # d2[i, j] = ||diff[i, j]||^2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    
    # Avoid division by zero
    # We only care about i < j
    # Mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d2_pairs = d2[mask]
    
    # Energy: sum log(1 / d) = - sum log(d) = -0.5 sum log(d2)
    # If d2 is very small, log is -inf. 
    # In practice, if points are distinct, it's fine.
    # We add a small epsilon to prevent log(0) if they get too close, 
    # but the gradient will push them apart.
    d2_safe = np.maximum(d2_pairs, 1e-12)
    energy = -0.5 * np.sum(np.log(d2_safe))
    
    # Gradient
    # We need dE/dtheta_i and dE/dphi_i
    # E = -0.5 * sum_{i<j} log(d2_ij)
    # dE/dtheta_i = -0.5 * sum_{j!=i} (1/d2_ij) * (d d2_ij / dtheta_i)
    # Note: sum over j!=i includes both (i,j) and (j,i) if we sum over all pairs, 
    # but our energy sum is over i<j.
    # Let's compute the gradient for each i by summing over all j != i.
    # Since d2_ij = d2_ji, the term for j in the sum for i is the same as the term for i in the sum for j.
    # So we can compute the full matrix of derivatives and sum.
    
    # d2_ij = 2 - 2 (sin_ti sin_tj cos(pi-pj) + cos_ti cos_tj)
    # Let c_ij = cos(phi_i - phi_j)
    # s_ij = sin(phi_i - phi_j)
    
    # Vectorized calculation of derivatives
    # We need d(d2_ij)/dtheta_i and d(d2_ij)/dphi_i for all i,j.
    
    # d2_ij = 2 - 2 (sin_ti sin_tj c_ij + cos_ti cos_tj)
    
    # d(d2_ij)/dtheta_i = -2 (cos_ti sin_tj c_ij - sin_ti cos_tj)
    # d(d2_ij)/dphi_i = -2 (sin_ti sin_tj (-s_ij)) = 2 sin_ti sin_tj s_ij
    
    # We compute these for all pairs i,j.
    # theta_i, theta_j shapes (N, 1) and (1, N)
    ti = theta[:, np.newaxis]
    tj = theta[np.newaxis, :]
    pi = phi[:, np.newaxis]
    pj = phi[np.newaxis, :]
    
    sin_ti = np.sin(ti)
    cos_ti = np.cos(ti)
    sin_tj = np.sin(tj)
    cos_tj = np.cos(tj)
    
    cos_diff = np.cos(pi - pj)
    sin_diff = np.sin(pi - pj)
    
    # d2 matrix
    d2_mat = 2 - 2 * (sin_ti * sin_tj * cos_diff + cos_ti * cos_tj)
    
    # Derivatives
    # d_d2_d_theta_i = -2 * (cos_ti * sin_tj * cos_diff - sin_ti * cos_tj)
    d_d2_d_theta_i = -2 * (cos_ti * sin_tj * cos_diff - sin_ti * cos_tj)
    # d_d2_d_phi_i = 2 * sin_ti * sin_tj * sin_diff
    d_d2_d_phi_i = 2 * sin_ti * sin_tj * sin_diff
    
    # Gradient components for each i
    # grad_theta[i] = -0.5 * sum_{j != i} (1/d2_ij) * d_d2_d_theta_i[i,j]
    # We can sum over all j, then subtract the i=i term (which is 0 anyway because d2_ii=0 is masked or handled)
    # Actually d2_ii = 0, so 1/d2 is inf. We must exclude i=j.
    
    # Mask diagonal
    mask_diag = ~np.eye(N, dtype=bool)
    
    # Safe d2 for division
    d2_safe_mat = np.where(mask_diag, d2_mat, 1.0) # Set diagonal to 1 to avoid div by 0, we'll mask later
    
    inv_d2 = np.where(mask_diag, 1.0 / d2_safe_mat, 0.0)
    
    grad_theta = -0.5 * np.sum(inv_d2 * d_d2_d_theta_i, axis=1)
    grad_phi = -0.5 * np.sum(inv_d2 * d_d2_d_phi_i, axis=1)
    
    grad = np.column_stack((grad_theta, grad_phi))
    
    return energy, grad

def _reconstruct(theta_phi):
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack((x, y, z))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_energy = np.inf
    best_pts = None
    
    # Initializations
    inits = []
    
    # 1. Fibonacci
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    # Convert to theta, phi
    theta_fib = np.arccos(z)
    phi_fib = np.arctan2(y, x)
    # Handle negative phi
    phi_fib = np.where(phi_fib < 0, phi_fib + 2*np.pi, phi_fib)
    inits.append(np.column_stack((theta_fib, phi_fib)))
    
    # 2. Random uniform on sphere
    np.random.seed(42)
    for k in range(5):
        v = np.random.randn(N, 3)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        theta_rand = np.arccos(v[:, 2])
        phi_rand = np.arctan2(v[:, 1], v[:, 0])
        phi_rand = np.where(phi_rand < 0, phi_rand + 2*np.pi, phi_rand)
        inits.append(np.column_stack((theta_rand, phi_rand)))
        
    # 3. Perturbed Fibonacci
    for k in range(5):
        theta_p = theta_fib + np.random.randn(N) * 0.1
        phi_p = phi_fib + np.random.randn(N) * 0.1
        theta_p = np.clip(theta_p, 1e-6, np.pi - 1e-6)
        phi_p = np.where(phi_p < 0, phi_p + 2*np.pi, phi_p)
        inits.append(np.column_stack((theta_p, phi_p)))
        
    # Optimize each initialization
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                _energy_and_grad, 
                init, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            energy = res.fun
            if energy < best_energy:
                best_energy = energy
                best_pts = _reconstruct(res.x)
        except Exception as e:
            pass # Ignore failures
            
    # Fallback to Fibonacci if no optimization succeeded
    if best_pts is None:
        best_pts = _reconstruct(inits[0])
        
    return best_pts
