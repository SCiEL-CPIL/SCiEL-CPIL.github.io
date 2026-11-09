"""Root finder for 2D systems using Broyden's method with robust fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """Find x in R^2 such that ||F(x)|| <= 1e-8."""
    start_time = time.time()
    DEADLINE = 115.0
    MAX_CALLS = 495
    
    x = np.array(x0, dtype=float)
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf
    
    def safe_F(x):
        nonlocal call_count
        if call_count >= MAX_CALLS:
            raise RuntimeError("Call budget exceeded")
        call_count += 1
        try:
            result = np.asarray(F(x), dtype=float).reshape(2)
        except Exception:
            return np.array([1e10, 1e10])
        if not np.all(np.isfinite(result)):
            return np.array([1e10, 1e10])
        return result
    
    # Initial evaluation
    Fx = safe_F(x)
    nrm = float(np.linalg.norm(Fx))
    best_x = x.copy()
    best_norm = nrm
    if nrm <= 1e-8:
        return x
    
    # Initial Jacobian via forward differences
    h = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (safe_F(x + e) - Fx) / h
    
    # Try to get inverse
    try:
        J_inv = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        J_inv = None
    
    # If Jacobian is singular, fall back to simple damped Newton
    if J_inv is None:
        for _ in range(100):
            if time.time() - start_time > DEADLINE or call_count >= MAX_CALLS:
                break
            Fx = safe_F(x)
            nrm = float(np.linalg.norm(Fx))
            if nrm < best_norm:
                best_x = x.copy()
                best_norm = nrm
            if nrm <= 1e-8:
                return x
            # Recompute J
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = 1e-7
                J[:, j] = (safe_F(x + e) - Fx) / 1e-7
            try:
                step = np.linalg.solve(J, Fx)
            except np.linalg.LinAlgError:
                break
            # Damped step
            alpha = 1.0
            for _ in range(10):
                x_new = x - alpha * step
                Fx_new = safe_F(x_new)
                nrm_new = float(np.linalg.norm(Fx_new))
                if nrm_new < best_norm:
                    best_x = x_new.copy()
                    best_norm = nrm_new
                if nrm_new <= 1e-8:
                    return x_new
                if nrm_new < nrm * 0.9:
                    x = x_new
                    Fx = Fx_new
                    nrm = nrm_new
                    break
                alpha *= 0.5
        return best_x
    
    # Broyden's method
    no_progress = 0
    for iteration in range(400):
        if time.time() - start_time > DEADLINE:
            break
        if call_count >= MAX_CALLS:
            break
        
        # Newton step
        step = J_inv @ Fx
        
        # Check step size
        step_norm = np.linalg.norm(step)
        if step_norm < 1e-15:
            # Try refreshing Jacobian
            if call_count + 2 < MAX_CALLS:
                h = 1e-7
                J = np.zeros((2, 2))
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h
                    J[:, j] = (safe_F(x + e) - Fx) / h
                try:
                    J_inv = np.linalg.inv(J)
                except np.linalg.LinAlgError:
                    break
            else:
                break
            continue
        
        # Backtracking line search
        alpha = 1.0
        accepted = False
        for k in range(8):
            x_new = x - alpha * step
            Fx_new = safe_F(x_new)
            nrm_new = float(np.linalg.norm(Fx_new))
            
            if nrm_new < best_norm:
                best_x = x_new.copy()
                best_norm = nrm_new
            
            if nrm_new <= 1e-8:
                return x_new
            
            if nrm_new < nrm:
                # Accept
                s = Fx_new - Fx
                delta_x = x_new - x
                denom = float(delta_x @ delta_x)
                if denom > 1e-16:
                    J = J + np.outer(s - J @ delta_x, delta_x) / denom
                    try:
                        J_inv = np.linalg.inv(J)
                    except np.linalg.LinAlgError:
                        break
                x = x_new
                Fx = Fx_new
                nrm = nrm_new
                accepted = True
                no_progress = 0
                break
            alpha *= 0.5
        
        if not accepted:
            no_progress += 1
            if no_progress >= 3:
                # Refresh Jacobian
                if call_count + 2 < MAX_CALLS and time.time() - start_time < DEADLINE:
                    h = 1e-7
                    J = np.zeros((2, 2))
                    for j in range(2):
                        e = np.zeros(2)
                        e[j] = h
                        J[:, j] = (safe_F(x + e) - Fx) / h
                    try:
                        J_inv = np.linalg.inv(J)
                    except np.linalg.LinAlgError:
                        break
                    no_progress = 0
                else:
                    break
        else:
            no_progress = 0
    
    return best_x
# EVOLVE-BLOCK-END
