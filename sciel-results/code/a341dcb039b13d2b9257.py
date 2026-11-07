"""Root finder for 3 fixed two-dimensional systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    x = np.array(x0, dtype=float)
    Fx = F(x)
    norm_F = np.linalg.norm(Fx)
    
    if norm_F <= 1e-8:
        return x
    
    best_x = x.copy()
    best_norm = norm_F
    
    # Adaptive step size for finite differences
    h = 1e-5
    max_iter = 100
    max_ls = 6
    
    # Threshold to switch to fast polish mode (pure Newton with minimal line search)
    polish_threshold = 1e-4
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Determine if we're in polish mode
        in_polish = norm_F < polish_threshold
        
        # Compute Jacobian using forward differences
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = F(x + e)
            J[:, j] = (Fx_e - Fx) / h
        
        # Try to solve J dx = -Fx
        try:
            dx = -np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # Regularize if singular
            try:
                dx = -np.linalg.solve(J + 1e-8 * np.eye(2), Fx)
            except np.linalg.LinAlgError:
                # Give up on Newton, try a gradient-like step
                dx = -Fx * 0.01
        
        # Limit step size to prevent wild jumps
        step_norm = np.linalg.norm(dx)
        if step_norm > 1.0:
            dx = dx / step_norm
        
        if in_polish:
            # In polish mode: accept full Newton step if it reduces norm,
            # otherwise try half step, then quarter. Max 3 evaluations.
            accepted = False
            for alpha in [1.0, 0.5, 0.25]:
                x_candidate = x + alpha * dx
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                
                if norm_candidate <= 1e-8:
                    return x_candidate
                
                if norm_candidate < best_norm:
                    best_x = x_candidate.copy()
                    best_norm = norm_candidate
                
                if norm_candidate < norm_F:
                    accepted = True
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    break
            
            if not accepted:
                # Even in polish mode, if Newton doesn't help, try gradient step
                x_candidate = x - 0.1 * Fx
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                
                if norm_candidate < norm_F:
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                else:
                    break
        else:
            # Normal mode: line search with bisection
            alpha = 1.0
            accepted = False
            for ls_iter in range(max_ls):
                x_candidate = x + alpha * dx
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                
                if norm_candidate <= 1e-8:
                    return x_candidate
                
                if norm_candidate < best_norm:
                    best_x = x_candidate.copy()
                    best_norm = norm_candidate
                
                # Accept if we made progress
                if norm_candidate < norm_F * 0.5:
                    accepted = True
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    break
                else:
                    alpha *= 0.5
            
            if not accepted:
                # Try a smaller step in the negative gradient direction
                alpha = 0.1
                x_candidate = x - alpha * Fx
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                
                if norm_candidate < norm_F:
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                else:
                    # Stuck, break
                    break
        
        # Check convergence
        if norm_F <= 1e-8:
            return x
        
        # Adaptive h: adjust based on function scale
        if norm_F > 1.0:
            h = 1e-4
        elif norm_F < 1e-3:
            h = 1e-6
    
    # Return the best solution found
    return best_x
# EVOLVE-BLOCK-END
