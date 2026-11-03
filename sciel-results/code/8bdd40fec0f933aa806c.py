import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin from 150s limit

    # 1. Define energy and gradient functions operating directly on Cartesian coordinates
    # This is simpler and often more robust than spherical coordinates for optimization.
    
    def compute_energy_and_grad(pts):
        # pts: (N, 3)
        n = pts.shape[0]
        # Ensure points are on sphere (project)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        
        # Compute pairwise squared distances
        # d2_ij = |x_i - x_j|^2 = 2 - 2 x_i . x_j
        dots = pts @ pts.T
        d2 = 2.0 - 2.0 * dots
        # Numerical stability: clip d2 to avoid log(0) or negative values
        np.fill_diagonal(d2, 1.0) # dummy
        d2 = np.maximum(d2, 1e-16)
        
        # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i<j} log(d2_ij)
        iu = np.triu_indices(n, 1)
        d2_u = d2[iu]
        energy = -0.5 * np.sum(np.log(d2_u))
        
        # Gradient w.r.t Cartesian coordinates
        # dE/dx_i = sum_{j!=i} (x_j - x_i) / d2_ij
        # Note: d/dx_i log(1/d_ij) = d/dx_i (-log d_ij) = - (1/d_ij) * (1/(2 d_ij)) * 2 (x_i - x_j) 
        #       = - (x_i - x_j) / d2_ij = (x_j - x_i) / d2_ij
        # Wait, let's re-verify.
        # E = sum_{i<j} -log(d_ij)
        # dE/dx_i = sum_{j!=i} - (1/d_ij) * (1/(2 d_ij)) * 2 (x_i - x_j) = - sum_{j!=i} (x_i - x_j) / d2_ij
        # So grad_x_i = - sum_{j!=i} (x_i - x_j) / d2_ij = sum_{j!=i} (x_j - x_i) / d2_ij
        
        inv_d2 = 1.0 / d2
        np.fill_diagonal(inv_d2, 0.0)
        
        # grad_x_i = sum_j x_j / d2_ij - x_i * sum_j 1/d2_ij
        sum_x_inv_d2 = pts.T @ inv_d2  # shape (3, N)
        sum_inv_d2 = np.sum(inv_d2, axis=1) # shape (N,)
        
        x_i_term = pts * sum_inv_d2[:, np.newaxis]
        grad_x = sum_x_inv_d2.T - x_i_term # shape (N, 3)
        
        # Project gradient onto tangent space of the sphere
        # For a point x on the sphere, the tangent space is orthogonal to x.
        # grad_tangent = grad_x - (grad_x . x) x
        dot = np.sum(grad_x * pts, axis=1, keepdims=True)
        grad_tangent = grad_x - dot * pts
        
        return energy, grad_tangent

    def compute_energy_only(pts):
        e, _ = compute_energy_and_grad(pts)
        return e

    def compute_precise_energy(pts):
        """High precision energy using math.fsum as per evaluator."""
        n = pts.shape[0]
        terms = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-9:
                    return float('inf')
                terms.append(math.log(1.0 / d))
        return math.fsum(terms)

    # 2. Initialization strategies
    
    candidates = []
    
    # Strategy A: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    pts_fib = np.column_stack([x, y, z])
    candidates.append(pts_fib)
    
    # Strategy B: Jittered Fibonacci
    rng = np.random.default_rng(42)
    for _ in range(5):
        if time.time() > deadline - 30: break
        pts_jit = pts_fib.copy()
        # Add small random perturbations
        noise = rng.standard_normal((N, 3)) * 0.1
        pts_jit = pts_fib + noise
        # Normalize
        norms = np.linalg.norm(pts_jit, axis=1, keepdims=True)
        pts_jit = pts_jit / norms
        candidates.append(pts_jit)
        
    # Strategy C: Random
    for _ in range(5):
        if time.time() > deadline - 30: break
        pts_r = rng.standard_normal((N, 3))
        norms = np.linalg.norm(pts_r, axis=1, keepdims=True)
        pts_r = pts_r / norms
        candidates.append(pts_r)

    best_energy = float('inf')
    best_points = None

    # 3. Optimization Loop
    # Use L-BFGS-B on Cartesian coordinates with projection? 
    # Actually, standard minimize doesn't handle manifold constraints directly.
    # We have two choices:
    # 1. Use spherical coordinates (theta, phi) as in the original.
    # 2. Use Cartesian coordinates and project after each step? No, minimize assumes Euclidean space.
    # 3. Use a custom line search or Riemannian optimizer.
    
    # The original program used spherical coordinates. This is a valid parameterization of the sphere (minus singularities).
    # Let's stick to spherical coordinates for the optimizer to avoid constraint handling issues, 
    # but we can also try using `minimize` with `method='L-BFGS-B'` on Cartesian coordinates 
    # and relying on the fact that the gradient is projected? No, that's not how it works.
    
    # Let's use the spherical coordinate approach from the original, as it is robust and proven.
    # However, we can improve the initialization and the final polish.
    
    def compute_points_spherical(theta, phi):
        st = np.sin(theta)
        ct = np.cos(theta)
        sp = np.sin(phi)
        cp = np.cos(phi)
        x = st * cp
        y = st * sp
        z = ct
        return np.column_stack([x, y, z])

    def energy_and_grad_spherical(params):
        theta = params[:N]
        phi = params[N:]
        
        st = np.sin(theta)
        ct = np.cos(theta)
        sp = np.sin(phi)
        cp = np.cos(phi)
        x = st * cp
        y = st * sp
        z = ct
        pts = np.column_stack([x, y, z])
        
        dots = pts @ pts.T
        d2 = 2.0 - 2.0 * dots
        np.fill_diagonal(d2, 1.0)
        d2 = np.maximum(d2, 1e-16)
        
        iu = np.triu_indices(N, 1)
        d2_u = d2[iu]
        energy = -0.5 * np.sum(np.log(d2_u))
        
        inv_d2 = 1.0 / d2
        np.fill_diagonal(inv_d2, 0.0)
        
        sum_x_inv_d2 = pts.T @ inv_d2
        sum_inv_d2 = np.sum(inv_d2, axis=1)
        
        x_i_term = pts * sum_inv_d2[:, np.newaxis]
        grad_x = sum_x_inv_d2.T - x_i_term
        
        dx_dtheta = np.column_stack([ct * cp, ct * sp, -st])
        grad_theta = np.sum(grad_x * dx_dtheta, axis=1)
        
        dx_dphi = np.column_stack([-st * sp, st * cp, np.zeros(N)])
        grad_phi = np.sum(grad_x * dx_dphi, axis=1)
        
        grad = np.concatenate([grad_theta, grad_phi])
        
        return energy, grad

    # Convert initial candidates to spherical coordinates
    def pts_to_spherical(pts):
        z = pts[:, 2]
        r_xy = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        phi = np.arctan2(pts[:, 1], pts[:, 0])
        return np.concatenate([theta, phi])

    spherical_candidates = []
    for p in candidates:
        sp = pts_to_spherical(p)
        # Ensure theta is within bounds for stability, though arccos handles [-1,1]
        sp[:N] = np.clip(sp[:N], 1e-6, np.pi - 1e-6)
        sp[N:] = np.mod(sp[N:], 2 * np.pi)
        spherical_candidates.append(sp)

    for p_init in spherical_candidates:
        if time.time() > deadline - 20:
            break
            
        try:
            res = minimize(
                energy_and_grad_spherical,
                p_init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if res.fun < best_energy:
                theta_opt = np.clip(res.x[:N], 1e-6, np.pi - 1e-6)
                phi_opt = np.mod(res.x[N:], 2 * np.pi)
                pts_opt = compute_points_spherical(theta_opt, phi_opt)
                
                # Verify with precise energy
                e_val = compute_precise_energy(pts_opt)
                
                if e_val < best_energy:
                    best_energy = e_val
                    best_points = pts_opt
                    best_params = np.concatenate([theta_opt, phi_opt])
                    
        except Exception:
            continue

    # 4. Final Polish
    # If we have time, do a few more iterations with tighter tolerance on the best solution
    if best_points is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(
                energy_and_grad_spherical,
                best_params,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100}
            )
            theta_final = np.clip(res_final.x[:N], 1e-6, np.pi - 1e-6)
            phi_final = np.mod(res_final.x[N:], 2 * np.pi)
            pts_final = compute_points_spherical(theta_final, phi_final)
            
            e_final = compute_precise_energy(pts_final)
            
            if e_final < best_energy:
                best_energy = e_final
                best_points = pts_final
        except Exception:
            pass

    if best_points is None:
        # Fallback to Fibonacci
        best_points = candidates[0]

    # Normalize just in case
    norms = np.linalg.norm(best_points, axis=1)
    best_points = best_points / norms[:, np.newaxis]

    return best_points
