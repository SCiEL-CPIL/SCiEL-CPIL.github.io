"""Root finder for 3 fixed two-dimensional systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Start with Newton's method using finite-difference Jacobian.
    2. If Newton fails or stagnates, use a damped/modified Newton with line search.
    3. If still failing, try a few random restarts with Newton.
    4. Track elapsed time and return best result before deadline.
    """
    x0 = np.array(x0, dtype=float)
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin
    
    best_x = x0.copy()
    best_norm = np.inf
    
    def update_best(x):
        nonlocal best_x, best_norm
        try:
            Fx = F(x)
            nrm = np.linalg.norm(Fx)
            if np.all(np.isfinite(Fx)) and nrm < best_norm:
                best_x = x.copy()
                best_norm = nrm
        except:
            pass
    
    def newton_step(x, Fx, J):
        """Perform one Newton step with damping if needed."""
        try:
            dx = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            return None, 0
        
        # Check if step is reasonable
        step_norm = np.linalg.norm(dx)
        if not np.isfinite(step_norm):
            return None, 0
        
        # Try full step first
        x_new = x - dx
        Fx_new = F(x_new)
        nrm_new = np.linalg.norm(Fx_new)
        
        if np.all(np.isfinite(Fx_new)) and nrm_new < np.linalg.norm(Fx):
            return x_new, 1
        
        # Try damped steps
        for lam in [0.5, 0.25, 0.1, 0.05, 0.01]:
            x_damped = x - lam * dx
            Fx_d = F(x_damped)
            nrm_d = np.linalg.norm(Fx_d)
            if np.all(np.isfinite(Fx_d)) and nrm_d < np.linalg.norm(Fx):
                return x_damped, lam
        
        return None, 0
    
    def compute_jacobian(x, Fx):
        """Compute Jacobian using central differences."""
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x + e) - F(x - e)) / (2 * h)
        return J
    
    # Phase 1: Newton's method from x0
    x = x0.copy()
    for it in range(80):
        if time.time() > deadline:
            break
        
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            break
        nrm = np.linalg.norm(Fx)
        if nrm <= 1e-12:
            return x
        
        update_best(x)
        
        J = compute_jacobian(x, Fx)
        if not np.all(np.isfinite(J)):
            break
        
        x_new, lam = newton_step(x, Fx, J)
        if x_new is None:
            break
        x = x_new
    
    # Phase 2: If not converged, try a few random restarts
    if best_norm > 1e-8:
        rng = np.random.default_rng(42)
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Random restart near x0 or near best_x
            if attempt % 2 == 0:
                x_try = x0 + rng.normal(0, 0.5, size=2)
            else:
                x_try = best_x + rng.normal(0, 0.5, size=2)
            
            # Newton from this point
            for it in range(50):
                if time.time() > deadline:
                    break
                
                Fx = F(x_try)
                if not np.all(np.isfinite(Fx)):
                    break
                nrm = np.linalg.norm(Fx)
                if nrm <= 1e-12:
                    return x_try
                
                update_best(x_try)
                
                J = compute_jacobian(x_try, Fx)
                if not np.all(np.isfinite(J)):
                    break
                
                x_new, lam = newton_step(x_try, Fx, J)
                if x_new is None:
                    break
                x_try = x_new
            
            if best_norm <= 1e-8:
                break
    
    # Return the best result found
    if best_norm <= 1e-8:
        return best_x
    
    # If we haven't found a good enough root, return best_x anyway
    # (it might still be within tolerance due to floating point)
    Fx_best = F(best_x)
    if np.all(np.isfinite(Fx_best)) and np.linalg.norm(Fx_best) <= 1e-8:
        return best_x
    
    return best_x
# EVOLVE-BLOCK-END
