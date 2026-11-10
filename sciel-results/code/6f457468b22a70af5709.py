"""Optimize logarithmic energy of 150 points on the unit sphere."""
import numpy as np
from scipy.optimize import minimize
import time

N = 150

def _spherical_to_cartesian(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _log_energy_and_grad(params):
    theta = params[:N]
    phi = params[N:]
    pts = _spherical_to_cartesian(theta, phi)
    
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, 1.0)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    iu = np.triu_indices(N, 1)
    d = dist[iu]
    # Clamp to avoid log(0)
    d = np.maximum(d, 1e-15)
    energy = -np.sum(np.log(d))
    
    # Vectorized Gradient
    # dE/dtheta_i = -sum_{j != i} (xi - xj)·(dxi/dtheta_i) / d_ij^2
    # dE/dphi_i   = -sum_{j != i} (xi - xj)·(dxi/dphi_i)   / d_ij^2
    
    # dxi/dtheta_i = (cos(theta_i)*cos(phi_i), cos(theta_i)*sin(phi_i), -sin(theta_i))
    # dxi/dphi_i   = (-sin(phi_i), cos(phi_i), 0)
    
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    
    # Shape: (N, 3)
    dxi_dtheta = np.stack([ct * cp, ct * sp, -st], axis=1)
    dxi_dphi = np.stack([-sp, cp, np.zeros(N)], axis=1)
    
    # We need to sum over j for each i.
    # Matrix M where M[i,j] = 1/d_ij^2 for i!=j, 0 otherwise.
    # We already have dist_sq.
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-30
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    # term_theta[i,j] = (xi - xj) dot dxi_dtheta_i
    # (xi - xj) is diff[i,j]. dxi_dtheta_i is same for all j.
    # We can compute this as:
    # dot_t[i,j] = sum_k (pts[i,k] - pts[j,k]) * dxi_dtheta[i,k]
    
    # Using broadcasting:
    # diff is (N, N, 3)
    # dxi_dtheta is (N, 3) -> (N, 1, 3)
    # dxi_dphi is (N, 3) -> (N, 1, 3)
    
    dot_t = np.sum(diff * dxi_dtheta[:, np.newaxis, :], axis=2)
    dot_p = np.sum(diff * dxi_dphi[:, np.newaxis, :], axis=2)
    
    # grad_theta[i] = - sum_j dot_t[i,j] * inv_dist_sq[i,j]
    # grad_phi[i]   = - sum_j dot_p[i,j] * inv_dist_sq[i,j]
    
    grad_theta = -np.sum(dot_t * inv_dist_sq, axis=1)
    grad_phi = -np.sum(dot_p * inv_dist_sq, axis=1)
    
    grad = np.concatenate([grad_theta, grad_phi])
    return energy, grad

def _log_energy_only(params):
    theta = params[:N]
    phi = params[N:]
    pts = _spherical_to_cartesian(theta, phi)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(N, 1)
    d = dist[iu]
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_energy = np.inf
    best_params = None
    
    # Initial point: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi_init = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    theta_init = np.arccos(np.clip(z, -1, 1))
    phi_init = np.mod(phi_init, 2 * np.pi)
    
    params0 = np.concatenate([theta_init, phi_init])
    
    # Try multiple restarts with different initializations
    # Since gradient is fast now, we can afford more iterations per restart
    # and potentially more restarts if time permits.
    n_restarts = 5
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            p0 = params0.copy()
        else:
            # Perturbed starting point
            noise = rng.normal(0, 0.1, size=2*N)
            p0 = params0.copy() + noise
            p0[:N] = np.clip(p0[:N], 1e-6, np.pi - 1e-6)
            p0[N:] = np.mod(p0[N:], 2 * np.pi)
        
        try:
            # Increased maxiter and tighter tolerances due to speedup
            result = minimize(
                _log_energy_and_grad,
                p0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            
            if result.success or result.fun < best_energy:
                e = _log_energy_only(result.x)
                if e < best_energy:
                    best_energy = e
                    best_params = result.x.copy()
        except Exception:
            continue
    
    # Final polish with tighter tolerances if we have time
    if best_params is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                _log_energy_and_grad,
                best_params,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 80}
            )
            e = _log_energy_only(result.x)
            if e < best_energy:
                best_energy = e
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        # Fallback to initial
        best_params = params0
    
    theta = best_params[:N]
    phi = best_params[N:]
    theta = np.clip(theta, 1e-8, np.pi - 1e-8)
    phi = np.mod(phi, 2 * np.pi)
    
    pts = _spherical_to_cartesian(theta, phi)
    
    # Normalize to unit sphere (safety)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    return pts
