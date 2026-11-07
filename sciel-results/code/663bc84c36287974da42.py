"""Root finder for 3 fixed two-dimensional systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    x = np.array(x0, dtype=float)
    Fx = F(x)
    norm_F = np.linalg.norm(Fx)
    
    if norm_F <= 1e-8:
        return x
    
    # Homotopy parameters
    lambda_curr = 0.0
    lambda_max = 1.0
    delta_lambda = 0.1
    min_delta_lambda = 1e-4
    
    # Store the target residual at x0
    F_x0 = Fx.copy()
    
    best_x = x.copy()
    best_norm = norm_F
    
    # Adaptive step size for finite differences
    h = 1e-5
    max_newton_iter = 10
    max_lambda_iter = 100
    
    # Helper to compute Jacobian
    def compute_jacobian(x_val, F_val):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = F(x_val + e)
            J[:, j] = (Fx_e - F_val) / h
        return J

    # Helper to perform a few Newton steps to solve F(x) = R
    def newton_solve(R, x_start, F_start):
        x_sol = x_start.copy()
        F_sol = F_start.copy()
        
        for _ in range(max_newton_iter):
            if time.time() > deadline:
                return x_sol, F_sol, False
            
            J = compute_jacobian(x_sol, F_sol)
            
            # Solve J dx = R - F_sol
            try:
                dx = np.linalg.solve(J, R - F_sol)
            except np.linalg.LinAlgError:
                # Regularize
                try:
                    dx = np.linalg.solve(J + 1e-8 * np.eye(2), R - F_sol)
                except np.linalg.LinAlgError:
                    return x_sol, F_sol, False
            
            # Damping
            alpha = 1.0
            accepted = False
            for _ in range(5):
                x_cand = x_sol + alpha * dx
                F_cand = F(x_cand)
                norm_cand = np.linalg.norm(F_cand - R)
                
                if norm_cand < np.linalg.norm(F_sol - R):
                    x_sol = x_cand
                    F_sol = F_cand
                    accepted = True
                    break
                else:
                    alpha *= 0.5
            
            if not accepted:
                break
                
            if np.linalg.norm(F_sol - R) < 1e-8:
                return x_sol, F_sol, True
        
        return x_sol, F_sol, False

    for _ in range(max_lambda_iter):
        if time.time() > deadline:
            break
            
        # Target residual for this lambda
        # H(x, lambda) = F(x) - (1-lambda)F(x0) = 0
        # So F(x) = (1-lambda)F(x0)
        
        # Try to advance lambda
        success = False
        current_delta = delta_lambda
        
        while current_delta >= min_delta_lambda:
            lambda_proposed = lambda_curr + current_delta
            if lambda_proposed > lambda_max:
                lambda_proposed = lambda_max
                current_delta = lambda_proposed - lambda_curr
            
            R_target = (1.0 - lambda_proposed) * F_x0
            
            # Solve for x such that F(x) = R_target
            x_new, F_new, conv = newton_solve(R_target, x, Fx)
            
            if conv:
                # Check if we actually improved the residual towards 0
                # The residual of the homotopy is F(x) - R_target
                # But we want F(x) to go to 0.
                # If lambda_proposed is close to 1, R_target is small.
                # We accept if the Newton solve converged.
                x = x_new
                Fx = F_new
                lambda_curr = lambda_proposed
                success = True
                break
            else:
                # Reduce step size
                current_delta *= 0.5
                # Optional: update x to the best guess from newton_solve even if not converged?
                # Usually better to stick to last known good point if step fails.
                
        if not success:
            # If we can't even take a tiny step, break
            break
            
        # Check if we've reached lambda_max and converged
        if lambda_curr >= lambda_max - 1e-6:
            if np.linalg.norm(Fx) <= 1e-8:
                return x
            # If lambda is 1 but norm is not small enough, do a final polish
            # This handles cases where the homotopy path ended at a point close to root but not quite
            for _ in range(20):
                if time.time() > deadline:
                    break
                J = compute_jacobian(x, Fx)
                try:
                    dx = -np.linalg.solve(J, Fx)
                except np.linalg.LinAlgError:
                    break
                
                alpha = 1.0
                accepted = False
                for _ in range(5):
                    x_cand = x + alpha * dx
                    F_cand = F(x_cand)
                    norm_cand = np.linalg.norm(F_cand)
                    
                    if norm_cand < np.linalg.norm(Fx):
                        x = x_cand
                        Fx = F_cand
                        accepted = True
                        break
                    else:
                        alpha *= 0.5
                if not accepted:
                    break
                if np.linalg.norm(Fx) <= 1e-8:
                    return x
            break
            
        # Update best solution
        if np.linalg.norm(Fx) < best_norm:
            best_x = x.copy()
            best_norm = np.linalg.norm(Fx)
            
        # If we are close to the end, increase step size to finish faster
        if lambda_curr > 0.5:
            delta_lambda = min(0.2, delta_lambda * 1.5)
        else:
            delta_lambda = 0.1

    return best_x
# EVOLVE-BLOCK-END
