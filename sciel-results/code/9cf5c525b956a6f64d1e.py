"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 using a multi-start Broyden's method:
    1. Try from x0 with Broyden's method (central-diff Jacobian init)
    2. If not converged, try from several perturbed starts
    3. Return the best result found
    
    Budget: <500 calls to F total across all attempts.
    """
    x0 = np.array(x0, dtype=float, copy=True)
    deadline = time.time() + 135.0  # safety margin before 150s kill
    call_count = 0
    best_x = x0.copy()
    best_norm = np.inf
    
    def _call_F(x_val):
        nonlocal call_count
        if time.time() > deadline:
            return None
        call_count += 1
        result = F(x_val)
        result = np.asarray(result, dtype=float)
        return result
    
    def _broyden_from(x_start, max_calls):
        """Run Broyden's method from x_start, using at most max_calls additional F evaluations."""
        nonlocal call_count, best_x, best_norm
        x = np.array(x_start, dtype=float, copy=True)
        
        Fx = _call_F(x)
        if Fx is None:
            return x, np.inf
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx < best_norm:
            best_x = x.copy()
            best_norm = norm_Fx
        if norm_Fx <= 1e-8:
            return x, norm_Fx
        
        # Central-difference Jacobian
        J = np.zeros((2, 2))
        h = 1e-6
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = _call_F(x + e)
            Fx_minus = _call_F(x - e)
            if Fx_plus is None or Fx_minus is None:
                # Forward difference fallback
                Fx_e = _call_F(x + e)
                if Fx_e is not None:
                    J[:, j] = (Fx_e - Fx) / h
                else:
                    J[:, j] = 1.0
            else:
                J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
        
        if not np.all(np.isfinite(J)) or np.linalg.cond(J) > 1e12:
            J = np.eye(2)
        
        # Broyden iterations
        for it in range(100):
            if time.time() > deadline:
                break
            if call_count >= max_calls:
                break
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.pinv(J) @ (-Fx)
                except:
                    break
            
            # Backtracking line search
            alpha = 1.0
            accepted = False
            for ls_iter in range(8):
                x_trial = x + alpha * dx
                if not np.all(np.isfinite(x_trial)):
                    alpha *= 0.5
                    continue
                
                Fx_trial = _call_F(x_trial)
                if Fx_trial is None:
                    break
                
                norm_trial = np.linalg.norm(Fx_trial)
                
                if norm_trial < best_norm:
                    best_x = x_trial.copy()
                    best_norm = norm_trial
                
                # Accept if we make progress (reduce norm) or if it's small enough
                if norm_trial <= norm_Fx * 0.95 or norm_trial <= 1e-8:
                    if norm_trial <= 1e-8:
                        return x_trial, norm_trial
                    s = x_trial - x
                    y = Fx_trial - Fx
                    s_norm_sq = s.T @ s
                    if s_norm_sq > 1e-30:
                        J = J + np.outer(y - J @ s, s.T) / s_norm_sq
                    x = x_trial
                    Fx = Fx_trial
                    norm_Fx = norm_trial
                    accepted = True
                    break
                else:
                    alpha *= 0.5
            
            if not accepted:
                break
        
        return x, norm_Fx
    
    # Attempt 1: From x0 with generous budget
    remaining = 500 - call_count
    _, _ = _broyden_from(x0, min(remaining, 300))
    
    if best_norm <= 1e-8:
        return best_x
    
    # If not converged, try a few perturbed starts
    perturbations = [
        x0 + 0.1 * np.array([1.0, 0.0]),
        x0 + 0.1 * np.array([0.0, 1.0]),
        x0 + 0.1 * np.array([1.0, 1.0]),
        x0 + 0.5 * np.array([1.0, -1.0]),
        x0 - 0.5 * np.array([1.0, 1.0]),
    ]
    
    for x_try in perturbations:
        if time.time() > deadline:
            break
        if call_count >= 490:
            break
        if best_norm <= 1e-8:
            break
        
        remaining = 490 - call_count
        if remaining < 10:
            break
        
        _, _ = _broyden_from(x_try, remaining)
    
    return best_x
# EVOLVE-BLOCK-END
