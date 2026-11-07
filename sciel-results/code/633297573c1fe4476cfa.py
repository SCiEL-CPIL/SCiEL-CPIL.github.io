"""Root finder for 3 fixed 2-D systems: multi-strategy with careful budget management."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Try Broyden's method with good initial Jacobian estimate.
    2. If that fails, try damped Newton with finite-difference Jacobian.
    3. If that fails, try a global search (grid + local refinement).
    4. Return best result found within budget.
    
    Budget:
    - Time limit: 140s internal deadline (150s hard limit).
    - Call limit: < 490 calls per system (leaves margin).
    """
    t_start = time.time()
    deadline = t_start + 140.0
    x = np.array(x0, dtype=float)
    calls = 0
    CALL_BUDGET = 490
    
    def time_ok():
        return time.time() < deadline
    
    def can_call():
        return calls < CALL_BUDGET and time_ok()
    
    def safe_F(x_val):
        nonlocal calls
        if not can_call():
            return None
        try:
            result = np.asarray(F(x_val), dtype=float)
            calls += 1
            if result.shape == (2,) and np.all(np.isfinite(result)):
                return result
            return None
        except Exception:
            calls += 1
            return None
    
    # Initial evaluation
    Fx = safe_F(x)
    if Fx is None:
        return x
    
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)
    
    if best_norm <= 1e-8:
        return best_x
    
    # Strategy 1: Broyden's method with finite-difference initial Jacobian
    result = _broyden(F, x, Fx, best_x, best_norm, safe_F, can_call, time_ok, deadline, CALL_BUDGET)
    if result is not None:
        return result
    
    # Update best from Broyden
    x = best_x.copy()
    Fx = safe_F(x)
    if Fx is None:
        return best_x
    best_norm = np.linalg.norm(Fx)
    if best_norm <= 1e-8:
        return best_x
    
    # Strategy 2: Damped Newton with finite-difference Jacobian
    result = _damped_newton(F, x, Fx, best_x, best_norm, safe_F, can_call, time_ok, deadline, CALL_BUDGET)
    if result is not None:
        return result
    
    x = best_x.copy()
    Fx = safe_F(x)
    if Fx is None:
        return best_x
    best_norm = np.linalg.norm(Fx)
    if best_norm <= 1e-8:
        return best_x
    
    # Strategy 3: Global grid search + local refinement
    result = _global_search(F, x0, best_x, best_norm, safe_F, can_call, time_ok, deadline, CALL_BUDGET)
    if result is not None:
        return result
    
    return best_x


def _broyden(F, x, Fx, best_x, best_norm, safe_F, can_call, time_ok, deadline, CALL_BUDGET):
    """Broyden's method with rank-1 update and trust-region damping."""
    x = x.copy()
    Fx = Fx.copy()
    best_x = best_x.copy()
    best_norm = best_norm
    
    # Get initial Jacobian via finite differences
    h = 1e-6
    J = np.eye(2)
    jac_valid = True
    for j in range(2):
        if not can_call():
            jac_valid = False
            break
        e = np.zeros(2)
        e[j] = h
        Fx_p = safe_F(x + e)
        if Fx_p is None:
            jac_valid = False
            break
        J[:, j] = (Fx_p - Fx) / h
    
    if not jac_valid:
        # Try with larger h
        h = 1e-5
        J = np.eye(2)
        jac_valid = True
        for j in range(2):
            if not can_call():
                jac_valid = False
                break
            e = np.zeros(2)
            e[j] = h
            Fx_p = safe_F(x + e)
            if Fx_p is None:
                jac_valid = False
                break
            J[:, j] = (Fx_p - Fx) / h
    
    if not jac_valid:
        return None
    
    # Check if Jacobian is well-conditioned
    try:
        cond = np.linalg.cond(J)
        if cond > 1e12:
            return None
    except np.linalg.LinAlgError:
        return None
    
    max_iter = 30
    for it in range(max_iter):
        if not can_call() or best_norm <= 1e-8:
            break
        
        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break
        
        if not np.all(np.isfinite(dx)):
            break
        
        # Trust-region: limit step size
        step_norm = np.linalg.norm(dx)
        if step_norm > 1.0:
            dx = dx * (1.0 / step_norm)
        
        # Try full step
        x_new = x + dx
        Fx_new = safe_F(x_new)
        
        if Fx_new is None:
            # Try half step
            dx = dx * 0.5
            x_new = x + dx
            Fx_new = safe_F(x_new)
            if Fx_new is None:
                break
        
        norm_new = np.linalg.norm(Fx_new)
        
        # Update best
        if norm_new < best_norm:
            best_x = x_new.copy()
            best_norm = norm_new
        
        if norm_new <= 1e-8:
            return best_x
        
        # Broyden rank-1 update
        y = Fx_new - Fx
        denom = dx @ dx
        if denom > 1e-14:
            J = J + np.outer(y - J @ dx, dx) / denom
        
        # Accept step if it reduces residual
        if norm_new < best_norm:
            x = x_new
            Fx = Fx_new
        else:
            # Try smaller step
            dx = dx * 0.5
            x_new = x + dx
            Fx_new = safe_F(x_new)
            if Fx_new is not None:
                norm_new = np.linalg.norm(Fx_new)
                if norm_new < best_norm:
                    x = x_new
                    Fx = Fx_new
                    best_x = x_new.copy()
                    best_norm = norm_new
                    if norm_new <= 1e-8:
                        return best_x
                    # Update Broyden
                    y = Fx_new - Fx
                    denom = dx @ dx
                    if denom > 1e-14:
                        J = J + np.outer(y - J @ dx, dx) / denom
                else:
                    break
            else:
                break
    
    if best_norm <= 1e-8:
        return best_x
    return None


def _damped_newton(F, x, Fx, best_x, best_norm, safe_F, can_call, time_ok, deadline, CALL_BUDGET):
    """Damped Newton with finite-difference Jacobian and backtracking line search."""
    x = x.copy()
    Fx = Fx.copy()
    best_x = best_x.copy()
    best_norm = best_norm
    
    max_iter = 15
    for it in range(max_iter):
        if not can_call() or best_norm <= 1e-8:
            break
        
        # Finite-difference Jacobian
        h = 1e-6
        J = np.zeros((2, 2))
        valid = True
        for j in range(2):
            if not can_call():
                valid = False
                break
            e = np.zeros(2)
            e[j] = h
            Fx_p = safe_F(x + e)
            if Fx_p is None:
                valid = False
                break
            J[:, j] = (Fx_p - Fx) / h
        
        if not valid:
            break
        
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break
        
        if not np.all(np.isfinite(dx)):
            break
        
        # Backtracking line search
        alpha = 1.0
        accepted = False
        for _ in range(8):
            if not can_call():
                break
            x_new = x + alpha * dx
            Fx_new = safe_F(x_new)
            if Fx_new is not None:
                norm_new = np.linalg.norm(Fx_new)
                if norm_new < best_norm:
                    x = x_new
                    Fx = Fx_new
                    best_x = x_new.copy()
                    best_norm = norm_new
                    if norm_new <= 1e-8:
                        return best_x
                    accepted = True
                    break
            alpha *= 0.5
        
        if not accepted:
            break
    
    if best_norm <= 1e-8:
        return best_x
    return None


def _global_search(F, x0, best_x, best_norm, safe_F, can_call, time_ok, deadline, CALL_BUDGET):
    """Global grid search around x0, then local refinement."""
    x0 = np.array(x0, dtype=float)
    best_x = best_x.copy()
    best_norm = best_norm
    
    # Grid search around x0
    # Try multiple scales
    scales = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
    found = None
    
    for scale in scales:
        if not can_call():
            break
        
        # Grid points around x0
        offsets = [
            np.array([0, 0]),
            np.array([scale, 0]), np.array([-scale, 0]),
            np.array([0, scale]), np.array([0, -scale]),
            np.array([scale, scale]), np.array([-scale, -scale]),
            np.array([scale, -scale]), np.array([-scale, scale]),
        ]
        
        for offset in offsets:
            if not can_call():
                break
            x_try = x0 + offset
            Fx_try = safe_F(x_try)
            if Fx_try is not None:
                norm_try = np.linalg.norm(Fx_try)
                if norm_try < best_norm:
                    best_x = x_try.copy()
                    best_norm = norm_try
                    if norm_try <= 1e-8:
                        return best_x
    
    # Local refinement from best point found
    if can_call():
        result = _broyden(F, best_x, safe_F(best_x), best_x, best_norm, safe_F, can_call, time_ok, deadline, CALL_BUDGET)
        if result is not None:
            return result
    
    if best_norm <= 1e-8:
        return best_x
    return None
# EVOLVE-BLOCK-END
