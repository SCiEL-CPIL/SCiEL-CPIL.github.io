import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Energy function (Cartesian)
    def energy(X):
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        dist = np.sqrt(dist_sq)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        energy_val = -np.sum(np.log(dist[mask]))
        return energy_val

    # Gradient function (Cartesian)
    def gradient(X):
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        return grad

    # Cartesian to Spherical (theta=azimuth, phi=polar)
    def cartesian_to_spherical(X):
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        theta = np.arctan2(y, x)  # [0, 2*pi)
        phi = np.arccos(np.clip(z, -1.0, 1.0))  # [0, pi]
        return np.column_stack([theta, phi])

    # Spherical to Cartesian
    def spherical_to_cartesian(sp):
        theta, phi = sp[:, 0], sp[:, 1]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.column_stack([x, y, z])

    # Energy and gradient in spherical coordinates
    def energy_and_grad_spherical(sp):
        X = spherical_to_cartesian(sp)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        dist = np.sqrt(dist_sq)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        e = -np.sum(np.log(dist[mask]))
        
        # Gradient in Cartesian
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        grad_cart = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Jacobian of spherical to Cartesian
        # dX/dtheta = [-sin(phi)*sin(theta), sin(phi)*cos(theta), 0]
        # dX/dphi = [cos(phi)*cos(theta), cos(phi)*sin(theta), -sin(phi)]
        theta, phi = sp[:, 0], sp[:, 1]
        sp_phi = np.sin(phi)
        cp_phi = np.cos(phi)
        st = np.sin(theta)
        ct = np.cos(theta)
        
        dX_dtheta = np.column_stack([-sp_phi * st, sp_phi * ct, np.zeros(N)])
        dX_dphi = np.column_stack([cp_phi * ct, cp_phi * st, -sp_phi])
        
        grad_theta = np.sum(grad_cart * dX_dtheta, axis=1)
        grad_phi = np.sum(grad_cart * dX_dphi, axis=1)
        
        return e, np.column_stack([grad_theta, grad_phi])

    def optimize_spherical(sp_init, time_limit):
        """Optimize in spherical coordinates using L-BFGS-B."""
        best_e, best_sp = energy_and_grad_spherical(sp_init)
        best_sp = sp_init.copy()
        
        t_start = time.time()
        
        # Use L-BFGS-B with time check
        try:
            def fun_and_grad(sp):
                if time.time() > time_limit:
                    return best_e, np.zeros_like(sp)
                e, g = energy_and_grad_spherical(sp)
                return e, g
            
            # Bounds: theta in [0, 2*pi], phi in [0, pi]
            bounds = [(0, 2 * np.pi), (0, np.pi)] * N
            
            result = minimize(
                fun_and_grad,
                sp_init,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if result.fun < best_e:
                best_e = result.fun
                best_sp = result.x.copy()
                
        except Exception:
            pass
        
        # If still have time, do a final polish with tighter tolerance
        if time.time() < time_limit - 2.0:
            try:
                def fun_and_grad2(sp):
                    if time.time() > time_limit:
                        return best_e, np.zeros_like(sp)
                    e, g = energy_and_grad_spherical(sp)
                    return e, g
                
                bounds = [(0, 2 * np.pi), (0, np.pi)] * N
                result2 = minimize(
                    fun_and_grad2,
                    best_sp,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
                )
                
                if result2.fun < best_e:
                    best_e = result2.fun
                    best_sp = result2.x.copy()
                    
            except Exception:
                pass
        
        return best_sp, best_e

    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    X_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_overall_X = X_fib
    best_overall_energy = energy(X_fib)
    
    # Try a few random initializations as well
    rng = np.random.default_rng(42)
    
    # Allocate time for restarts
    num_restarts = 6
    time_per_restart = (deadline - time.time()) / num_restarts
    
    # First restart with Fibonacci
    sp_fib = cartesian_to_spherical(X_fib)
    time_limit = time.time() + time_per_restart
    sp_best, e_best = optimize_spherical(sp_fib, time_limit)
    X_best = spherical_to_cartesian(sp_best)
    # Ensure points are on unit sphere
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    e_best_check = energy(X_best)
    if e_best_check < best_overall_energy:
        best_overall_energy = e_best_check
        best_overall_X = X_best

    for k in range(num_restarts - 1):
        if time.time() > deadline - 1.0:
            break
        
        # Random initialization
        X_rand = rng.normal(size=(N, 3))
        X_rand = X_rand / np.linalg.norm(X_rand, axis=1, keepdims=True)
        
        time_per_restart = (deadline - time.time()) / (num_restarts - k)
        time_limit = time.time() + time_per_restart
        
        sp_rand = cartesian_to_spherical(X_rand)
        sp_best, e_best = optimize_spherical(sp_rand, time_limit)
        X_best = spherical_to_cartesian(sp_best)
        X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
        e_best_check = energy(X_best)
        if e_best_check < best_overall_energy:
            best_overall_energy = e_best_check
            best_overall_X = X_best

    # Final high-precision polish on the best configuration
    if time.time() < deadline - 3.0:
        sp_best = cartesian_to_spherical(best_overall_X)
        time_limit = deadline - 1.0
        sp_polished, e_polished = optimize_spherical(sp_best, time_limit)
        X_polished = spherical_to_cartesian(sp_polished)
        X_polished = X_polished / np.linalg.norm(X_polished, axis=1, keepdims=True)
        e_polished_check = energy(X_polished)
        if e_polished_check < best_overall_energy:
            best_overall_energy = e_polished_check
            best_overall_X = X_polished

    return best_overall_X
