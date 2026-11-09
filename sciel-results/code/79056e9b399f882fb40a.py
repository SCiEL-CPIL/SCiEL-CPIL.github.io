"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def find_root_system(F, x0):
    """
    Finds a root x in R^2 for F(x) = 0.
    Strategy: 
    1. Global Phase: Use Nelder-Mead to minimize ||F(x)||^2. This is derivative-free 
       and helps navigate away from singularities or local minima of the residual norm 
       that might stall a direct Newton method. It uses a bounded number of calls.
    2. Local Phase: Use Broyden's Method starting from the best point found in Phase 1.
       Broyden's method approximates the Jacobian and updates it using rank-1 updates,
       requiring only 1 function evaluation per iteration (after the initial Jacobian 
       estimation). This is significantly cheaper than Newton (which needs ~3-5 evals 
       per iteration for FD Jacobian) and allows for more iterations within the budget 
       to reach the tight 1e-8 tolerance.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def time_left():
        return deadline - time.time()
        
    def objective(x):
        """Objective for Nelder-Mead: ||F(x)||^2"""
        try:
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return 1e20
            return np.dot(Fx, Fx)
        except Exception:
            return 1e20

    def norm(vec):
        return np.linalg.norm(vec)

    def is_good(x):
        if not np.all(np.isfinite(x)):
            return False
        try:
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return False
            return norm(Fx) <= 1e-8
        except Exception:
            return False

    def broyden_solver(F, x0, max_iter=100, tol=1e-12):
        """
        Broyden's method for finding a root.
        Requires 1 F call per iteration after the first.
        """
        x = np.array(x0, dtype=float)
        try:
            Fx = F(x)
        except Exception:
            return x
            
        if not np.all(np.isfinite(Fx)):
            return x
            
        # Initialize Jacobian with finite differences (Central difference for better accuracy)
        # This costs 4 F calls (2 for x, 2 for y) if we use central, or 3 if forward.
        # Let's use forward difference to save 1 call, or central if budget allows.
        # Given we are in the local phase, accuracy is good. Let's use forward.
        eps = 1e-5
        J = np.zeros((2, 2))
        
        # Check if we have enough time for the Jacobian setup
        if time_left() < 0.5:
             return x
             
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            try:
                Fx_e = F(x + e)
                if not np.all(np.isfinite(Fx_e)):
                    return x
                J[:, j] = (Fx_e - Fx) / eps
            except Exception:
                return x
                
        if not np.all(np.isfinite(J)):
            return x

        best_x = x.copy()
        best_norm = norm(Fx)
        
        for i in range(max_iter):
            if time_left() < 0.5:
                break
                
            current_norm = best_norm
            if current_norm <= tol:
                return best_x

            # Solve J dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # If singular, perturb J or break
                # Try a small perturbation to J
                J += 1e-6 * np.eye(2)
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    break
                    
            if not np.all(np.isfinite(dx)):
                break

            # Damping / Line Search
            # We want to ensure ||F(x + lambda dx)|| < ||F(x)||
            step_found = False
            lam = 1.0
            for _ in range(10):
                x_new = x + lam * dx
                try:
                    F_new = F(x_new)
                except Exception:
                    lam *= 0.5
                    continue
                    
                if not np.all(np.isfinite(F_new)):
                    lam *= 0.5
                    continue
                    
                new_norm = norm(F_new)
                
                if new_norm < current_norm:
                    # Update Broyden
                    # s = x_new - x
                    # y = F_new - F
                    # J_new = J + (y - J s) s^T / (s^T s)
                    s = x_new - x
                    y = F_new - Fx
                    denom = np.dot(s, s)
                    if denom > 1e-12:
                        J += (y - J.dot(s)) * np.outer(s, s) / denom
                    
                    x = x_new
                    Fx = F_new
                    step_found = True
                    break
                else:
                    lam *= 0.5
            
            if not step_found:
                break
                
            current_norm = norm(Fx)
            if current_norm < best_norm:
                best_norm = current_norm
                best_x = x.copy()
                
            # Convergence check on step
            if norm(dx) < 1e-10:
                break
                
        return best_x

    # --- Main Logic ---
    
    x0 = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0)):
        x0 = np.array([0.0, 0.0])
        
    # Phase 1: Global Search using Nelder-Mead
    # We limit the number of function evaluations to keep the budget low.
    # Nelder-Mead in 2D is efficient.
    # options: maxiter, maxfev
    # Let's allow up to 100 evaluations for the global phase.
    # This is cheap compared to Newton iterations which might stall.
    
    try:
        res = minimize(
            objective, 
            x0, 
            method='Nelder-Mead',
            options={'maxiter': 100, 'maxfev': 100, 'xatol': 1e-5, 'fatol': 1e-10}
        )
        x_start_broyden = res.x
    except Exception:
        x_start_broyden = x0

    # Check if Nelder-Mead already found a good root (unlikely to be 1e-8 but possible)
    if is_good(x_start_broyden):
        return x_start_broyden

    # Phase 2: Local Polish using Broyden's Method
    # Start from the best point found by Nelder-Mead
    x_result = broyden_solver(F, x_start_broyden)
    
    if is_good(x_result):
        return x_result

    # Fallback: If Broyden from Nelder-Mead result failed, try Broyden from x0 directly
    # This handles cases where Nelder-Mead wandered into a bad region or converged slowly
    if is_good(x0):
         return x0
         
    x_result_2 = broyden_solver(F, x0)
    
    if is_good(x_result_2):
        return x_result_2

    # Return the best of the two results based on residual norm
    try:
        n1 = norm(F(x_result))
        n2 = norm(F(x_result_2))
        if n1 <= n2:
            return x_result
        else:
            return x_result_2
    except Exception:
        return x_result

# EVOLVE-BLOCK-END
