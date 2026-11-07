"""Root finder for 3 fixed 2D systems using Broyden's method with line search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Robust root finder for 2D systems using Broyden's method with line search.
    
    F: callable, takes array of shape (2,), returns array of shape (2,)
    x0: initial guess, array of shape (2,)
    Returns: x such that ||F(x)|| <= 1e-8
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    def time_up():
        return time.time() > deadline
    
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        return x
    
    # Check initial point
    try:
        Fx = F(x)
    except:
        return x
    if not np.all(np.isfinite(Fx)):
        return x
    norm_F = np.linalg.norm(Fx)
    if norm_F <= 1e-8:
        return x
    
    # Estimate initial Jacobian via finite differences
    h = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        try:
            Fx_plus = F(x + e)
            if not np.all(np.isfinite(Fx_plus)):
                Fx_plus = Fx
        except:
            Fx_plus = Fx
        J[:, j] = (Fx_plus - Fx) / h
    
    # Check if J is singular
    try:
        det = np.linalg.det(J)
        if abs(det) < 1e-12:
            J += 1e-10 * np.eye(2)
    except:
        J += 1e-10 * np.eye(2)
    
    best_x = x.copy()
    best_norm = norm_F
    
    # Main iteration loop
    for iteration in range(100):
        if time_up():
            break
        
        # Try damped Newton step
        try:
            direction = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If Newton fails, try a steepest descent step
            direction = -Fx
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1.0:
                direction = direction / dir_norm
        
        if not np.all(np.isfinite(direction)):
            break
        
        # Line search with backtracking
        step_size = 1.0
        max_backtrack = 15
        found_better = False
        
        for backtrack in range(max_backtrack):
            if time_up():
                break
            
            new_x = x + step_size * direction
            
            if not np.all(np.isfinite(new_x)):
                step_size *= 0.5
                continue
            
            try:
                new_Fx = F(new_x)
            except:
                step_size *= 0.5
                continue
            
            if not np.all(np.isfinite(new_Fx)):
                step_size *= 0.5
                continue
            
            new_norm = np.linalg.norm(new_Fx)
            
            if new_norm <= 1e-8:
                best_x = new_x
                best_norm = new_norm
                return best_x
            
            # Armijo condition for line search
            if new_norm < norm_F:
                # Update Broyden's formula
                s = step_size * direction
                y = new_Fx - Fx
                s_dot_s = np.dot(s, s)
                if s_dot_s > 1e-20:
                    J = J + (y - J @ s) @ s.T / s_dot_s
                
                x = new_x
                Fx = new_Fx
                norm_F = new_norm
                found_better = True
                break
            else:
                step_size *= 0.5
        
        if found_better:
            if norm_F <= best_norm:
                best_x = x.copy()
                best_norm = norm_F
        else:
            # No progress made with current Jacobian
            # Try restarting with fresh finite difference Jacobian
            h = max(1e-8, 0.1 * np.linalg.norm(x))
            J_new = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                try:
                    Fx_plus = F(x + e)
                    if not np.all(np.isfinite(Fx_plus)):
                        Fx_plus = Fx
                except:
                    Fx_plus = Fx
                J_new[:, j] = (Fx_plus - Fx) / h
            
            # Check if new J is better conditioned
            try:
                cond_old = np.linalg.cond(J)
            except:
                cond_old = np.inf
            try:
                cond_new = np.linalg.cond(J_new)
            except:
                cond_new = np.inf
            
            if cond_new < cond_old or np.isinf(cond_old):
                J = J_new
            else:
                # Add regularization
                J = J + 1e-8 * np.eye(2)
        
        # Check for convergence
        if norm_F <= 1e-8:
            best_x = x.copy()
            best_norm = norm_F
            return best_x
        
        # Check if x is getting too large
        if np.linalg.norm(x) > 1e10:
            break
    
    # Return best found
    if best_norm <= norm_F:
        return best_x
    return x
# EVOLVE-BLOCK-END
