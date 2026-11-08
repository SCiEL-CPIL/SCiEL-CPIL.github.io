import numpy as np
import time

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _is_finite(x):
    return np.all(np.isfinite(x)) and x.shape == (2,)

def _norm(x):
    return float(np.linalg.norm(x))

def _safe_call(F, x):
    """Call F safely, returning None on failure."""
    try:
        y = F(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float)
        if y.shape != (2,):
            return None
        if not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None

def find_root_system(F, x0):
    global _START_TIME
    _START_TIME = time.time()
    deadline = _START_TIME + 110.0  # Safety margin from 120s limit
    
    x = np.array(x0, dtype=float)
    
    # Initial evaluation
    Fx = _safe_call(F, x)
    if Fx is None:
        # If initial point is invalid, try a small perturbation
        x = x + np.array([0.1, 0.1])
        Fx = _safe_call(F, x)
        if Fx is None:
            return x

    if _norm(Fx) <= 1e-8:
        return x

    # Broyden's method
    # We need an initial Jacobian estimate. We use finite differences for the first step.
    # This costs 2 extra calls (total 3 for setup).
    # Subsequent steps cost 1 call each.
    
    h = 1e-6
    e1 = np.array([h, 0.0])
    e2 = np.array([0.0, h])
    
    Fx1 = _safe_call(F, x + e1)
    Fx2 = _safe_call(F, x + e2)
    
    if Fx1 is None or Fx2 is None:
        # Fallback to simple Newton if setup fails? Or just return best.
        # Let's try a few more points or just return.
        # For robustness, if we can't even compute the initial Jacobian, we are in trouble.
        # Try a slightly larger h?
        h = 1e-4
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        Fx1 = _safe_call(F, x + e1)
        Fx2 = _safe_call(F, x + e2)
        if Fx1 is None or Fx2 is None:
            return x

    J = np.zeros((2, 2))
    J[:, 0] = (Fx1 - Fx) / h
    J[:, 1] = (Fx2 - Fx) / h

    x_old = x
    Fx_old = Fx
    best_x = x
    best_norm = _norm(Fx)

    max_iter = 100
    
    for i in range(max_iter):
        if _elapsed() > deadline:
            break
            
        # Solve J * d = -Fx
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If J is singular, regularize it or break
            # Simple regularization: J + lambda * I
            lambda_reg = 1e-8
            J_reg = J + lambda_reg * np.eye(2)
            try:
                d = np.linalg.solve(J_reg, -Fx)
            except np.linalg.LinAlgError:
                break

        if not np.all(np.isfinite(d)):
            break
            
        # Line search for step size alpha
        alpha = 1.0
        norm_current = _norm(Fx)
        found_step = False
        
        # We want to minimize ||F(x + alpha*d)||^2
        # Simple backtracking
        for _ls in range(20):
            if _elapsed() > deadline:
                break
                
            x_new = x + alpha * d
            Fx_new = _safe_call(F, x_new)
            
            if Fx_new is None:
                alpha *= 0.5
                continue
                
            norm_new = _norm(Fx_new)
            
            if norm_new < norm_current:
                # Accept step
                x = x_new
                Fx = Fx_new
                found_step = True
                break
            else:
                alpha *= 0.5
                
        if not found_step:
            # If no step reduced the norm, we are stuck.
            # Try a small perturbation?
            # Or break.
            # If we can't make progress, break and return best.
            break
            
        # Update best
        if norm_new < best_norm:
            best_norm = norm_new
            best_x = x.copy()
            
        if best_norm <= 1e-8:
            break

        # Broyden Update
        # s = x_new - x_old
        # y = Fx_new - Fx_old
        s = x - x_old
        y = Fx - Fx_old
        
        # Avoid division by zero
        if np.dot(y, s) == 0:
            # If y is orthogonal to s, the update might be problematic.
            # Standard Broyden update: J = J + (y - J*s) * s^T / (s^T * s)
            # If s^T * s is 0, we didn't move. But we moved because norm decreased.
            # Let's check s^T * s
            s_dot_s = np.dot(s, s)
            if s_dot_s < 1e-16:
                break
            else:
                # Update
                J = J + np.outer(y - J @ s, s) / s_dot_s
        else:
            s_dot_s = np.dot(s, s)
            if s_dot_s < 1e-16:
                break
            else:
                # Update
                J = J + np.outer(y - J @ s, s) / s_dot_s

        x_old = x
        Fx_old = Fx

    return best_x
