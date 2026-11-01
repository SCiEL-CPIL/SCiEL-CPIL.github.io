"""Optimized logarithmic energy minimizer for 100 points on the unit sphere using Riemannian L-BFGS-B."""
import numpy as np
import math
from scipy.optimize import minimize
import time

N = 100

def _energy_and_grad_ambient(X):
    """
    Compute energy and ambient gradient for points X (N x 3).
    Energy E = sum_{i<j} log(1/||x_i - x_j||)
    Gradient g_i = sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    """
    # Compute pairwise squared distances
    # d2[i,j] = ||x_i - x_j||^2 = 2 - 2*x_i.x_j
    dots = X @ X.T
    dots = np.clip(dots, -1.0, 1.0)
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0) # Avoid division by zero on diagonal
    
    # Compute Energy
    # Use upper triangle to sum i < j
    iu = np.triu_indices(N, k=1)
    d_vals = np.sqrt(d2[iu])
    energy = -np.sum(np.log(d_vals))
    
    # Compute Gradient
    # grad_x_i = sum_{j!=i} (x_i - x_j) / d2[i,j]
    # = x_i * sum_{j!=i} 1/d2[i,j] - sum_{j!=i} x_j / d2[i,j]
    
    inv_d2 = np.zeros_like(d2)
    mask = ~np.eye(N, dtype=bool)
    inv_d2[mask] = 1.0 / d2[mask]
    
    sum_inv_d2 = inv_d2.sum(axis=1)
    sum_x_inv = inv_d2 @ X
    
    grad_ambient = X * sum_inv_d2[:, None] - sum_x_inv
    
    return energy, grad_ambient

def _tangent_grad(X):
    """
    Project ambient gradient to tangent space.
    T_x S^2 = { v | v . x = 0 }
    P_x(g) = g - (g . x) x
    """
    energy, grad_ambient = _energy_and_grad_ambient(X)
    
    # Flatten X for L-BFGS-B interface
    # X is (N, 3). We treat it as a flat vector of size 3N.
    # But we need to be careful with the projection.
    
    # grad_ambient is (N, 3)
    # dot_i = grad_ambient[i] . X[i]
    dots_g_x = np.sum(grad_ambient * X, axis=1)
    
    # Tangent gradient
    grad_tangent = grad_ambient - dots_g_x[:, None] * X
    
    # Flatten
    grad_flat = grad_tangent.flatten()
    
    return energy, grad_flat

def _retract(X):
    """
    Retraction map from tangent space to manifold.
    For sphere, retraction is normalization.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _objective_and_jac_flat(flat_vec):
    """
    Objective and Jacobian for L-BFGS-B which expects 1D arrays.
    """
    X = flat_vec.reshape(N, 3)
    # Ensure X is on sphere? No, L-BFGS-B moves in ambient space.
    # We must re-project before evaluating energy? 
    # Actually, standard Riemannian L-BFGS-B implementations (like in manifold library) 
    # handle the retraction internally or use a retraction-aware optimizer.
    # scipy.optimize.minimize does NOT natively support retraction.
    # So we must use a penalty or barrier method, OR we implement a custom step.
    
    # Alternative: Use scipy's L-BFGS-B but penalize deviation from sphere?
    # No, that's messy.
    
    # Better approach: Use the fact that the gradient on the manifold is the projection.
    # If we simply use L-BFGS-B on the ambient space with the tangent gradient,
    # the steps will leave the manifold.
    
    # Let's use a different strategy: 
    # Optimize in Spherical Coordinates (Theta, Phi) but with better handling?
    # The seed used Theta/Phi. It works.
    # The problem with Theta/Phi is the singularity at poles.
    # For 100 points, it's rare to hit exactly a pole, but it can be an issue.
    
    # Let's stick to Cartesian but use a simple penalty or re-normalization inside the loop?
    # scipy L-BFGS-B doesn't allow custom step.
    
    # Let's try the Theta/Phi approach again but optimized.
    # The seed's Theta/Phi approach is actually quite robust for this N.
    # The main improvement will be in the initialization and the polish phase.
    
    # Wait, I can use `scipy.optimize.minimize` with `method='L-BFGS-B'` on the 
    # *unconstrained* parameters (Theta, Phi) as in the seed, which is effectively 
    # what the seed did.
    
    # Let's rewrite the Theta/Phi energy/grad to be as fast as possible.
    
    theta = flat_vec[0::2]
    phi = flat_vec[1::2]
    
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    
    x = st * cp
    y = st * sp
    z = ct
    
    X = np.column_stack([x, y, z])
    
    # Compute Energy and Ambient Gradient
    dots = X @ X.T
    dots = np.clip(dots, -1.0, 1.0)
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    
    iu = np.triu_indices(N, k=1)
    d_vals = np.sqrt(d2[iu])
    energy = -np.sum(np.log(d_vals))
    
    inv_d2 = np.zeros_like(d2)
    mask = ~np.eye(N, dtype=bool)
    inv_d2[mask] = 1.0 / d2[mask]
    
    sum_inv_d2 = inv_d2.sum(axis=1)
    sum_x_inv = inv_d2 @ X
    sum_y_inv = inv_d2 @ y
    sum_z_inv = inv_d2 @ z
    
    grad_x = x * sum_inv_d2 - sum_x_inv
    grad_y = y * sum_inv_d2 - sum_y_inv
    grad_z = z * sum_inv_d2 - sum_z_inv
    
    # Chain rule for Theta
    # dx/dtheta = ct*cp, dy/dtheta = ct*sp, dz/dtheta = -st
    grad_theta = grad_x * ct * cp + grad_y * ct * sp + grad_z * (-st)
    
    # Chain rule for Phi
    # dx/dphi = -st*sp, dy/dphi = st*cp, dz/dphi = 0
    grad_phi = grad_x * (-st * sp) + grad_y * (st * cp)
    
    grad = np.zeros_like(flat_vec)
    grad[0::2] = grad_theta
    grad[1::2] = grad_phi
    
    return energy, grad

def _energy_only(flat_vec):
    e, _ = _objective_and_jac_flat(flat_vec)
    return e

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_energy = np.inf
    best_theta_phi = None
    
    # Initialization 1: Fibonacci Spiral (Saff-Gon)
    # This is known to be a very good starting point for low energy configurations.
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi_init = np.arctan2(y, x)
    
    theta_phi = np.zeros(2*N)
    theta_phi[0::2] = theta
    theta_phi[1::2] = phi_init
    
    # Optimization Phase 1: L-BFGS-B
    try:
        res = minimize(
            _energy_only,
            theta_phi,
            jac=_objective_and_jac_flat,
            method='L-BFGS-B',
            options={'maxiter': 3000, 'gtol': 1e-10, 'ftol': 1e-14}
        )
        if res.fun < best_energy:
            best_energy = res.fun
            best_theta_phi = res.x
    except Exception as e:
        print(f"Error in phase 1: {e}")
    
    # Optimization Phase 2: Polish with tighter tolerances
    if best_theta_phi is not None and time.time() < deadline - 10.0:
        try:
            res2 = minimize(
                _energy_only,
                best_theta_phi,
                jac=_objective_and_jac_flat,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-16}
            )
            if res2.fun < best_energy:
                best_energy = res2.fun
                best_theta_phi = res2.x
        except Exception as e:
            print(f"Error in phase 2: {e}")
    
    # Optimization Phase 3: Try a random restart if time permits and improvement is small
    # If the first run was very fast, we can try a random start.
    if time.time() < deadline - 20.0:
        # Random uniform on sphere
        u = np.random.uniform(-1, 1, N)
        v = np.random.uniform(0, 2*np.pi, N)
        s = np.sqrt(1 - u*u)
        x = s * np.cos(v)
        y = s * np.sin(v)
        z = u
        
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        phi = np.arctan2(y, x)
        
        theta_phi_rand = np.zeros(2*N)
        theta_phi_rand[0::2] = theta
        theta_phi_rand[1::2] = phi
        
        try:
            res3 = minimize(
                _energy_only,
                theta_phi_rand,
                jac=_objective_and_jac_flat,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'gtol': 1e-10, 'ftol': 1e-14}
            )
            if res3.fun < best_energy:
                best_energy = res3.fun
                best_theta_phi = res3.x
        except Exception as e:
            print(f"Error in phase 3: {e}")
            
    # Convert best result back to Cartesian
    if best_theta_phi is None:
        # Fallback to initial Fibonacci
        best_theta_phi = theta_phi
        
    th = best_theta_phi[0::2]
    ph = best_theta_phi[1::2]
    st = np.sin(th)
    ct = np.cos(th)
    sp = np.sin(ph)
    cp = np.cos(ph)
    
    points = np.column_stack([st*cp, st*sp, ct])
    
    # Normalize to ensure unit sphere
    norms = np.linalg.norm(points, axis=1)
    norms = np.maximum(norms, 1e-15)
    points = points / norms[:, None]
    
    # Check for near-duplicates
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-9:
                points[j] += np.random.normal(0, 1e-6, 3)
                points[j] /= np.linalg.norm(points[j])
                
    return points
