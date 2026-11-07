"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    x = np.array(x0, dtype=float)
    Fx = F(x)
    norm_F = np.linalg.norm(Fx)
    
    # If already close, return
    if norm_F <= 1e-8:
        return x

    # Use central differences for better Jacobian accuracy
    h = 1e-6
    lambda_lm = 1e-8
    max_iter = 100
    max_ls = 10
    deadline = start_time + 115  # Safety margin before 120s limit
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break

        # Compute Jacobian using central differences
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x + e) - F(x - e)) / (2 * h)
        
        # Try to solve J dx = -Fx (Newton step)
        dx = None
        # Use Levenberg-Marquardt regularization if J is ill-conditioned
        for _lm_iter in range(10):
            try:
                J_reg = J + lambda_lm * np.eye(2)
                dx = -np.linalg.solve(J_reg, Fx)
                break
            except np.linalg.LinAlgError:
                lambda_lm *= 100
        
        if dx is None:
            break
        
        # Line search with backtracking
        alpha = 1.0
        step = dx
        accepted = False
        x_new = x
        F_new = Fx
        norm_new = norm_F
        
        for _ in range(max_ls):
            if time.time() > deadline:
                break
            x_candidate = x + alpha * step
            F_candidate = F(x_candidate)
            norm_candidate = np.linalg.norm(F_candidate)
            
            if norm_candidate <= 1e-8:
                return x_candidate
            
            # Check for improvement
            if norm_candidate < norm_F:
                accepted = True
                x_new = x_candidate
                F_new = F_candidate
                norm_new = norm_candidate
                break
            else:
                alpha *= 0.5
        
        if not accepted:
            break
            
        x = x_new
        Fx = F_new
        norm_F = norm_new

    # Final polish: if we're close but not quite there, do a few more iterations with tighter tolerance
    if norm_F < 1e-4 and time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline:
                break
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = 1e-8
                J[:, j] = (F(x + e) - F(x - e)) / (2 * 1e-8)
            
            try:
                dx = -np.linalg.solve(J, Fx)
                x_candidate = x + dx
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                if norm_candidate < norm_F:
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    if norm_F <= 1e-8:
                        return x
            except np.linalg.LinAlgError:
                break

    return x
# EVOLVE-BLOCK-END
