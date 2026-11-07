# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with adaptive line search and fallback strategies.
    """
    x = np.array(x0, dtype=float).reshape(2)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_up():
        return time.time() > deadline
    
    def norm(v):
        return np.linalg.norm(v)
    
    def check_convergence(x, Fx):
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(Fx)):
            return False
        return norm(Fx) <= 1e-8
    
    # Initial evaluation
    Fx = F(x)
    if check_convergence(x, Fx):
        return x
    
    best_x = x.copy()
    best_Fx = Fx.copy()
    best_norm = norm(Fx)
    
    # Initialize Jacobian estimate
    J = np.eye(2)
    # Try to get a better initial Jacobian using finite differences
    h = 1e-7
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        try:
            Fx_plus = F(x + e)
            if np.all(np.isfinite(Fx_plus)):
                J[:, j] = (Fx_plus - Fx) / h
        except:
            pass
    
    # Broyden's method with line search
    for iteration in range(200):
        if time_up():
            break
        
        # Check convergence
        if norm(Fx) <= 1e-8:
            return x
        
        # Update best if current is better
        if norm(Fx) < best_norm:
            best_x = x.copy()
            best_Fx = Fx.copy()
            best_norm = norm(Fx)
        
        # Try to solve J * delta = -Fx
        try:
            delta = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If J is singular, use pseudo-inverse
            try:
                delta = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                break
        
        # Line search
        alpha = 1.0
        Fx_norm = norm(Fx)
        
        for ls_iter in range(20):
            if time_up():
                break
            
            x_trial = x + alpha * delta
            try:
                Fx_trial = F(x_trial)
            except:
                alpha *= 0.5
                continue
            
            if not np.all(np.isfinite(Fx_trial)):
                alpha *= 0.5
                continue
            
            Fx_trial_norm = norm(Fx_trial)
            
            # Armijo condition
            if Fx_trial_norm <= Fx_norm - 1e-4 * alpha * np.dot(delta, Fx):
                break
            
            alpha *= 0.5
        
        else:
            # If line search failed, try a smaller step or break
            if alpha < 1e-12:
                break
            x_trial = x + alpha * delta
            try:
                Fx_trial = F(x_trial)
            except:
                break
            if not np.all(np.isfinite(Fx_trial)):
                break
        
        # Check convergence
        if norm(Fx_trial) <= 1e-8:
            return x_trial
        
        # Update Broyden's formula
        s = x_trial - x
        y = Fx_trial - Fx
        if norm(s) > 1e-15:
            # Broyden's update: J = J + (y - J*s) * s^T / (s^T * s)
            J = J + np.outer(y - J @ s, s) / np.dot(s, s)
        
        x = x_trial
        Fx = Fx_trial
    
    # If Broyden didn't converge, try a few more strategies
    
    # Strategy 2: Try Newton with finite difference Jacobian from best_x
    x = best_x.copy()
    Fx = best_Fx.copy()
    
    for iteration in range(50):
        if time_up():
            break
        
        if norm(Fx) <= 1e-8:
            return x
        
        # Compute Jacobian via finite differences
        h = 1e-6
        J_fd = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            try:
                Fx_plus = F(x + e)
                if np.all(np.isfinite(Fx_plus)):
                    J_fd[:, j] = (Fx_plus - Fx) / h
                else:
                    # Try backward difference
                    Fx_minus = F(x - e)
                    if np.all(np.isfinite(Fx_minus)):
                        J_fd[:, j] = (Fx - Fx_minus) / h
                    else:
                        J_fd[:, j] = 1.0
            except:
                J_fd[:, j] = 1.0
        
        try:
            delta = np.linalg.solve(J_fd, -Fx)
        except:
            break
        
        # Line search
        alpha = 1.0
        Fx_norm = norm(Fx)
        
        for ls_iter in range(10):
            if time_up():
                break
            
            x_trial = x + alpha * delta
            try:
                Fx_trial = F(x_trial)
            except:
                alpha *= 0.5
                continue
            
            if not np.all(np.isfinite(Fx_trial)):
                alpha *= 0.5
                continue
            
            if norm(Fx_trial) <= Fx_norm * 0.5:
                break
            
            alpha *= 0.5
        
        if alpha < 1e-12:
            break
        
        x_trial = x + alpha * delta
        try:
            Fx_trial = F(x_trial)
        except:
            break
        
        if not np.all(np.isfinite(Fx_trial)):
            break
        
        if norm(Fx_trial) <= 1e-8:
            return x_trial
        
        x = x_trial
        Fx = Fx_trial
    
    # Return the best result found
    if np.all(np.isfinite(best_x)) and best_norm <= 1e-8:
        return best_x
    
    # If we didn't get below 1e-8, return the best we found
    # but first check if current x is better
    if np.all(np.isfinite(x)) and norm(Fx) < best_norm:
        return x
    
    return best_x
# EVOLVE-BLOCK-END
