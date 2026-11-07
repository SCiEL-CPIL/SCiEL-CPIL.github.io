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
    polish_threshold = 1e-3
    
    # Tolerance for convergence check
    tol = 1e-8
    
    total_calls = 1  # initial F(x0)
    
    def update_best(x_cand, norm_cand):
        nonlocal best_x, best_norm
        if norm_cand < best_norm:
            best_x = x_cand.copy()
            best_norm = norm_cand
    
    # Main Newton loop with adaptive polish/damped modes
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Determine if we're in polish mode
        in_polish = norm_F < polish_threshold
        
        # Compute Jacobian using central differences for better accuracy
        if in_polish:
            h_jac = min(h, 1e-6)  # Smaller step for polish mode
        else:
            h_jac = h
            
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h_jac
            Fx_e = F(x + e)
            J[:, j] = (Fx_e - Fx) / h_jac
            total_calls += 1
        
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
        
        if step_norm < 1e-15:
            break
        
        # If in polish mode and we have a good Newton step, try full step first
        if in_polish and norm_F < 1e-6:
            x_candidate = x + dx
            F_candidate = F(x_candidate)
            norm_candidate = np.linalg.norm(F_candidate)
            total_calls += 1
            
            if norm_candidate <= tol:
                return x_candidate
            
            update_best(x_candidate, norm_candidate)
            
            if norm_candidate < norm_F:
                x = x_candidate
                Fx = F_candidate
                norm_F = norm_candidate
                continue
        
        if in_polish:
            # In polish mode: accept full Newton step if it reduces norm,
            # otherwise try half step, then quarter. Max 3 evaluations.
            accepted = False
            for alpha in [1.0, 0.5, 0.25]:
                x_candidate = x + alpha * dx
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                total_calls += 1
                
                if norm_candidate <= tol:
                    return x_candidate
                
                update_best(x_candidate, norm_candidate)
                
                if norm_candidate < norm_F:
                    accepted = True
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    break
            
            if not accepted:
                # Stuck in polish mode, try a small gradient step
                x_candidate = x - 0.1 * Fx
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                total_calls += 1
                
                if norm_candidate < norm_F:
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    update_best(x, norm_F)
                else:
                    break
        else:
            # DAMPED MODE: Standard line search
            alpha = 1.0
            accepted = False
            for ls_iter in range(max_ls):
                x_candidate = x + alpha * dx
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                total_calls += 1
                
                if norm_candidate <= tol:
                    return x_candidate
                
                update_best(x_candidate, norm_candidate)
                
                # Accept if we made significant progress
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
                x_candidate = x - 0.05 * Fx
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                total_calls += 1
                
                if norm_candidate < norm_F:
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    update_best(x, norm_F)
                else:
                    # Stuck, break to try restarts
                    break
        
        # Check convergence
        if norm_F <= tol:
            return x
        
        # Adaptive h: adjust based on function scale
        if norm_F > 1.0:
            h = 1e-4
        elif norm_F < 1e-3:
            h = 1e-6
    
    # If main loop didn't converge, try multi-start restarts
    if best_norm > tol:
        radii = [0.5, 1.0, 2.0, 5.0]
        angles = [i * np.pi / 4 for i in range(8)]
        
        for r in radii:
            for angle in angles:
                if time.time() > deadline:
                    break
                if total_calls > 450:
                    break
                
                x_start = x0 + r * np.array([np.cos(angle), np.sin(angle)])
                try:
                    F_start = F(x_start)
                    total_calls += 1
                except Exception:
                    continue
                
                norm_start = np.linalg.norm(F_start)
                if norm_start <= tol:
                    return x_start
                update_best(x_start, norm_start)
                
                # Quick Newton from this start
                x_curr = x_start.copy()
                Fx_curr = F_start.copy()
                norm_curr = norm_start
                h_local = 1e-5
                
                for _ in range(15):
                    if time.time() > deadline:
                        break
                    if total_calls > 480:
                        break
                    
                    in_polish_local = norm_curr < polish_threshold
                    
                    # Use smaller h in polish mode
                    h_jac_local = min(h_local, 1e-6) if in_polish_local else h_local
                    
                    J = np.zeros((2, 2))
                    for j in range(2):
                        e = np.zeros(2)
                        e[j] = h_jac_local
                        Fx_e = F(x_curr + e)
                        J[:, j] = (Fx_e - Fx_curr) / h_jac_local
                        total_calls += 1
                    
                    try:
                        dx = -np.linalg.solve(J, Fx_curr)
                    except np.linalg.LinAlgError:
                        try:
                            dx = -np.linalg.solve(J + 1e-8 * np.eye(2), Fx_curr)
                        except np.linalg.LinAlgError:
                            dx = -Fx_curr * 0.01
                    
                    step_norm = np.linalg.norm(dx)
                    if step_norm > 1.0:
                        dx = dx / step_norm
                    if step_norm < 1e-15:
                        break
                    
                    # If in polish mode and we have a good Newton step, try full step first
                    if in_polish_local and norm_curr < 1e-6:
                        x_candidate = x_curr + dx
                        F_candidate = F(x_candidate)
                        norm_candidate = np.linalg.norm(F_candidate)
                        total_calls += 1
                        
                        if norm_candidate <= tol:
                            return x_candidate
                        
                        update_best(x_candidate, norm_candidate)
                        
                        if norm_candidate < norm_curr:
                            x_curr = x_candidate
                            Fx_curr = F_candidate
                            norm_curr = norm_candidate
                            continue
                    
                    if in_polish_local:
                        accepted = False
                        for alpha in [1.0, 0.5]:
                            x_candidate = x_curr + alpha * dx
                            F_candidate = F(x_candidate)
                            norm_candidate = np.linalg.norm(F_candidate)
                            total_calls += 1
                            
                            if norm_candidate <= tol:
                                return x_candidate
                            update_best(x_candidate, norm_candidate)
                            
                            if norm_candidate < norm_curr:
                                accepted = True
                                x_curr = x_candidate
                                Fx_curr = F_candidate
                                norm_curr = norm_candidate
                                break
                        if not accepted:
                            break
                    else:
                        alpha = 1.0
                        accepted = False
                        for _ in range(4):
                            x_candidate = x_curr + alpha * dx
                            F_candidate = F(x_candidate)
                            norm_candidate = np.linalg.norm(F_candidate)
                            total_calls += 1
                            
                            if norm_candidate <= tol:
                                return x_candidate
                            update_best(x_candidate, norm_candidate)
                            
                            if norm_candidate < norm_curr * 0.5:
                                accepted = True
                                x_curr = x_candidate
                                Fx_curr = F_candidate
                                norm_curr = norm_candidate
                                break
                            else:
                                alpha *= 0.5
                        if not accepted:
                            break
                    
                    if norm_curr <= tol:
                        return x_curr
                    
                    if norm_curr > 1.0:
                        h_local = 1e-4
                    elif norm_curr < 1e-3:
                        h_local = 1e-6
    
    return best_x
# EVOLVE-BLOCK-END
