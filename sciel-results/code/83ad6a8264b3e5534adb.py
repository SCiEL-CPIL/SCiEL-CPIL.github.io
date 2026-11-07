import numpy as np
import time

def find_root_system(F, x0):
    x = np.array(x0, dtype=float)
    best_x = x.copy()
    best_norm = float('inf')
    
    start_time = time.time()
    # The evaluator kills after 150s. We need a safety margin.
    # The final check is not counted, but our internal checks are.
    deadline = start_time + 140.0 
    max_calls = 450  # Safety margin below 500

    calls = 0

    def safe_F(x):
        nonlocal calls
        if time.time() > deadline:
            raise TimeoutError
        if calls >= max_calls:
            raise RuntimeError("Call limit exceeded")
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                return None
            calls += 1
            return Fx
        except Exception:
            return None

    def is_valid(x):
        return np.all(np.isfinite(x)) and x.shape == (2,)

    def update_best(x, norm):
        nonlocal best_x, best_norm
        if norm < best_norm:
            best_norm = norm
            best_x = x.copy()

    # Initial evaluation
    Fx = safe_F(x)
    if Fx is None:
        return best_x
    norm = np.linalg.norm(Fx)
    update_best(x, norm)
    if norm <= 1e-8:
        return best_x

    def broyden(F, x0, Fx0, max_iter=50):
        nonlocal calls, best_x, best_norm
        x = x0.copy()
        Fx = Fx0.copy()
        norm = np.linalg.norm(Fx)
        
        if norm <= 1e-8:
            return x, norm, True

        # Compute initial Jacobian using forward differences
        h = 1e-6
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_pert = safe_F(x + e)
            if Fx_pert is None:
                return x, norm, False
            J[:, j] = (Fx_pert - Fx) / h
        
        # Broyden's method
        for _ in range(max_iter):
            if time.time() > deadline or calls >= max_calls - 5:
                break
                
            try:
                # Solve J @ dx = -Fx
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except Exception:
                    break
            
            if not np.all(np.isfinite(dx)):
                break
            
            # Backtracking line search
            alpha = 1.0
            step_accepted = False
            Fx_old_norm = norm
            
            for _ in range(20):
                x_try = x + alpha * dx
                if not is_valid(x_try):
                    alpha *= 0.5
                    continue
                
                Fx_try = safe_F(x_try)
                if Fx_try is None:
                    alpha *= 0.5
                    continue
                
                norm_try = np.linalg.norm(Fx_try)
                update_best(x_try, norm_try)
                
                if norm_try < Fx_old_norm * 0.9:  # Sufficient decrease
                    # Broyden update
                    s = x_try - x
                    y = Fx_try - Fx
                    J_s = J @ s
                    diff = y - J_s
                    denom = s @ s
                    
                    if denom > 1e-15:
                        J = J + np.outer(diff, s) / denom
                    else:
                        # Fallback to finite diff Jacobian if s is too small
                        h = 1e-6
                        for j in range(2):
                            e = np.zeros(2)
                            e[j] = h
                            Fx_pert = safe_F(x_try + e)
                            if Fx_pert is None:
                                return x_try, norm_try, norm_try <= 1e-8
                            J[:, j] = (Fx_pert - Fx_try) / h
                    
                    x = x_try
                    Fx = Fx_try
                    norm = norm_try
                    step_accepted = True
                    break
                else:
                    alpha *= 0.5
            
            if not step_accepted:
                break
            
            if norm <= 1e-8:
                return x, norm, True
        
        return x, norm, norm <= 1e-8

    # Try Broyden from initial point
    x_result, norm_result, success = broyden(F, x, Fx, max_iter=30)
    if success:
        return x_result
    
    # If failed, try a few other starting points
    # Generate candidates
    candidates = []
    # Simple perturbations
    scales = [0.1, 0.5, 1.0, 2.0]
    for scale in scales:
        for sign in [-1, 1]:
            for axis in range(2):
                x_cand = x0.copy()
                x_cand[axis] += sign * scale * (abs(x0[axis]) + 1.0)
                candidates.append(x_cand)
    
    # Try each candidate
    for x_cand in candidates:
        if time.time() > deadline - 10 or calls >= max_calls - 10:
            break
        
        if not is_valid(x_cand):
            continue
        
        Fx_cand = safe_F(x_cand)
        if Fx_cand is None:
            continue
        
        norm_cand = np.linalg.norm(Fx_cand)
        update_best(x_cand, norm_cand)
        
        if norm_cand <= 1e-8:
            return x_cand
        
        # Try Broyden from this candidate with fewer iterations to save calls
        x_result, norm_result, success = broyden(F, x_cand, Fx_cand, max_iter=15)
        if success:
            return x_result

    return best_x
