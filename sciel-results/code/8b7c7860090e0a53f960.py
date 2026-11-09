"""Root finder for 3 fixed two-dimensional systems using a robust hybrid approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    # Deadline guard: return best result found if we're getting close to the limit
    start_time = time.time()
    deadline = 150.0  # seconds, with safety margin

    def time_ok():
        return (time.time() - start_time) < deadline

    def norm(x):
        return np.linalg.norm(x)

    # Try to get a good initial point and Jacobian estimate
    x = np.array(x0, dtype=float).copy()
    Fx = F(x)
    Fx = np.asarray(Fx, dtype=float)
    
    # If we're already at a root, return immediately
    if norm(Fx) <= 1e-8:
        return x

    best_x = x.copy()
    best_norm = norm(Fx)

    # Phase 1: Try Newton's method with finite-difference Jacobian
    # Use a few iterations to see if it converges quickly
    max_newton_iters = 20
    for it in range(max_newton_iters):
        if not time_ok():
            break
        if norm(Fx) <= 1e-8:
            return x
        
        # Compute finite-difference Jacobian
        J = np.zeros((2, 2))
        h = 1e-7
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = np.asarray(F(x + e), dtype=float)
            J[:, j] = (Fx_plus - Fx) / h
        
        # Try to solve the linear system
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break
        
        # Check if the step is reasonable
        if not np.all(np.isfinite(dx)):
            break
        
        # Damped Newton step
        step_size = 1.0
        x_trial = x + step_size * dx
        Fx_trial = np.asarray(F(x_trial), dtype=float)
        
        # Backtracking line search
        for _ in range(10):
            if np.all(np.isfinite(Fx_trial)) and norm(Fx_trial) < norm(Fx):
                break
            step_size *= 0.5
            x_trial = x + step_size * dx
            Fx_trial = np.asarray(F(x_trial), dtype=float)
        
        if not np.all(np.isfinite(Fx_trial)):
            break
        
        x = x_trial
        Fx = Fx_trial
        if norm(Fx) < best_norm:
            best_norm = norm(Fx)
            best_x = x.copy()
    
    if norm(Fx) <= 1e-8:
        return x

    # Phase 2: If Newton didn't converge, try Broyden's method
    # Initialize Jacobian with finite differences
    x = best_x.copy()
    Fx = np.asarray(F(x), dtype=float)
    best_norm = norm(Fx)
    
    J = np.zeros((2, 2))
    h = 1e-7
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_plus = np.asarray(F(x + e), dtype=float)
        J[:, j] = (Fx_plus - Fx) / h

    max_broyden_iters = 30
    for it in range(max_broyden_iters):
        if not time_ok():
            break
        if norm(Fx) <= 1e-8:
            return x
        
        # Solve for Newton direction
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Reset Jacobian to identity and try again
            J = np.eye(2)
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
        
        if not np.all(np.isfinite(dx)):
            break
        
        # Damped step with backtracking
        step_size = 1.0
        x_trial = x + step_size * dx
        Fx_trial = np.asarray(F(x_trial), dtype=float)
        
        for _ in range(10):
            if np.all(np.isfinite(Fx_trial)) and norm(Fx_trial) < norm(Fx) * (1 - 1e-4 * step_size):
                break
            step_size *= 0.5
            x_trial = x + step_size * dx
            Fx_trial = np.asarray(F(x_trial), dtype=float)
        
        if not np.all(np.isfinite(Fx_trial)):
            break
        
        # Broyden rank-1 update
        s = x_trial - x
        y = Fx_trial - Fx
        denom = np.dot(y, s)
        if abs(denom) > 1e-14:
            J = J + np.outer(y - J @ s, s) / denom
        
        x = x_trial
        Fx = Fx_trial
        if norm(Fx) < best_norm:
            best_norm = norm(Fx)
            best_x = x.copy()
    
    if norm(Fx) <= 1e-8:
        return x

    # Phase 3: If still not converged, do a final polish with tight Newton
    x = best_x.copy()
    Fx = np.asarray(F(x), dtype=float)
    
    if norm(Fx) > 1e-8:
        # Try a few more Newton steps with tighter tolerance
        for it in range(10):
            if not time_ok():
                break
            if norm(Fx) <= 1e-8:
                return x
            
            J = np.zeros((2, 2))
            h = 1e-8
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_plus = np.asarray(F(x + e), dtype=float)
                J[:, j] = (Fx_plus - Fx) / h
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            
            if not np.all(np.isfinite(dx)):
                break
            
            x_trial = x + dx
            Fx_trial = np.asarray(F(x_trial), dtype=float)
            
            if np.all(np.isfinite(Fx_trial)) and norm(Fx_trial) < norm(Fx):
                x = x_trial
                Fx = Fx_trial
    
    # Return the best result found
    if norm(Fx) <= 1e-8:
        return x
    return best_x
# EVOLVE-BLOCK-END
