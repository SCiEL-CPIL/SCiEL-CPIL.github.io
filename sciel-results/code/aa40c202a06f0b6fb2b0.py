"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    pts: (N, 3) array
    Returns: (energy, gradient)
    """
    N = pts.shape[0]
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    dist = np.sqrt(dist_sq)  # (N, N)
    
    # Mask diagonal to avoid division by zero
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: sum_{i<j} log(1/d_{ij}) = -sum_{i<j} log(d_{ij})
    # Use upper triangle only
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: d/dx_i [-log(d_{ij})] = (x_i - x_j) / d_{ij}^2
    # For each point i, grad_i = sum_{j != i} (x_i - x_j) / d_{ij}^2
    # But we need to project onto the sphere tangent space
    grad = np.zeros_like(pts)
    for i in range(N):
        for j in range(N):
            if i != j:
                d = dist[i, j]
                if d > 1e-12:
                    grad[i] += (pts[i] - pts[j]) / (d * d)
    
    # Project gradient onto tangent space: grad_tan = grad - (grad . x_i) * x_i
    for i in range(N):
        dot = np.dot(grad[i], pts[i])
        grad[i] -= dot * pts[i]
    
    return energy, grad


def construct():
    # Golden-angle (Fibonacci) spiral points on the unit sphere: a good but improvable start.
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalise
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, np.newaxis]
    
    # Time limit: 150s, keep safety margin of 10s
    start_time = time.time()
    time_limit = 150.0
    safety_margin = 10.0
    
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    
    # Try to improve with gradient-based optimization
    # Use L-BFGS-B with spherical constraint via parameterization or projection
    
    def _spherical_energy_and_grad(params):
        """Unpack spherical coordinates, compute energy and gradient.
        params: (2N,) array of (theta_i, phi_i) for each point
        """
        thetas = params[0::2]
        phis = params[1::2]
        
        # Convert to Cartesian
        x = np.sin(thetas) * np.cos(phis)
        y = np.sin(thetas) * np.sin(phis)
        z = np.cos(thetas)
        pts = np.column_stack([x, y, z])
        
        # Compute energy
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist = np.sqrt(dist_sq)
        iu = np.triu_indices(N, k=1)
        energy = -np.sum(np.log(dist[iu] + 1e-15))
        
        # Gradient in Cartesian
        grad_cart = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i != j:
                    d = dist[i, j]
                    if d > 1e-12:
                        grad_cart[i] += (pts[i] - pts[j]) / (d * d)
        
        # Convert gradient to spherical coordinates
        # dE/dtheta = grad_cart . dpts/dtheta
        # dE/dphi = grad_cart . dpts/dphi
        grad_params = np.zeros_like(params)
        for i in range(N):
            # dpts/dtheta = (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))
            dtheta = np.array([np.cos(thetas[i]) * np.cos(phis[i]),
                               np.cos(thetas[i]) * np.sin(phis[i]),
                               -np.sin(thetas[i])])
            # dpts/dphi = (-sin(theta)sin(phi), sin(theta)cos(phi), 0)
            dphi = np.array([-np.sin(thetas[i]) * np.sin(phis[i]),
                             np.sin(thetas[i]) * np.cos(phis[i]),
                             0.0])
            grad_params[2*i] = np.dot(grad_cart[i], dtheta)
            grad_params[2*i+1] = np.dot(grad_cart[i], dphi)
        
        return energy, grad_params
    
    # Initial spherical coordinates
    thetas = np.arccos(np.clip(pts[:, 2], -1.0, 1.0))
    phis = np.arctan2(pts[:, 1], pts[:, 0])
    params0 = np.zeros(2 * N)
    params0[0::2] = thetas
    params0[1::2] = phis
    
    # Optimize
    try:
        result = minimize(_spherical_energy_and_grad, params0, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'maxfun': 50000,
                                            'ftol': 1e-12, 'gtol': 1e-10})
        
        elapsed = time.time() - start_time
        if elapsed < time_limit - safety_margin:
            if result.success or result.fun < best_energy:
                thetas_opt = result.x[0::2]
                phis_opt = result.x[1::2]
                x = np.sin(thetas_opt) * np.cos(phis_opt)
                y = np.sin(thetas_opt) * np.sin(phis_opt)
                z = np.cos(thetas_opt)
                pts_opt = np.column_stack([x, y, z])
                norms = np.linalg.norm(pts_opt, axis=1)
                pts_opt = pts_opt / norms[:, np.newaxis]
                
                # Verify and update if better
                e_opt = _energy_and_grad(pts_opt)[0]
                if e_opt < best_energy:
                    best_pts = pts_opt
                    best_energy = e_opt
    except Exception:
        pass
    
    # Check time and return best result
    elapsed = time.time() - start_time
    if elapsed > time_limit - safety_margin:
        return best_pts
    
    return best_pts
# EVOLVE-BLOCK-END
