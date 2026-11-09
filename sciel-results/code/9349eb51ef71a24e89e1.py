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
    calls = 1
    if Fx is None:
        # If initial point is invalid, try a small perturbation
        x = x + np.array([0.1, 0.1])
        Fx = _safe_call(F, x)
        calls += 1
        if Fx is None:
            return x

    if _norm(Fx) <= 1e-12:
        return x

    # Initialize Jacobian with finite differences
    # This costs 2 extra calls (total 3 so far)
    h = 1e-5
    e1 = np.array([h, 0.0])
    e2 = np.array([0.0, h])
    
    Fx1 = _safe_call(F, x + e1)
    Fx2 = _safe_call(F, x + e2)
    calls += 2
    
    if Fx1 is None or Fx2 is None:
        # Fallback: If we can't even get FD Jacobian, try a very simple gradient descent on ||F||^2?
        # Or just return current best. Given the constraints, returning best is safer than guessing.
        # But let's try a small step in -Fx direction as a last resort? No, that's not a root finder.
        # Let's try to restart with a slightly different x?
        # For now, break and return.
        return x

    J = np.zeros((2, 2))
    J[:, 0] = (Fx1 - Fx) / h
    J[:, 1] = (Fx2 - Fx) / h

    max_iter = 490 # Leave buffer for safety
    max_ls_iter = 20
    
    # Broyden's Good Method
    # State: x, Fx, J
    # Update: J_new = J + (s - J*s) * s^T / (s^T * s) where s = x_new - x, and we assume F_new is known.
    # Actually, Broyden's update for the inverse Jacobian or direct Jacobian?
    # Direct Jacobian update (Good):
    # J_{k+1} = J_k + (F_{k+1} - F_k - J_k s_k) s_k^T / (s_k^T s_k)
    
    for i in range(max_iter):
        if _elapsed() > deadline or calls >= 495:
            break
            
        # Solve J * d = -Fx
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If J is singular, perturb it or break
            # Try adding small identity
            try:
                d = np.linalg.solve(J + 1e-12 * np.eye(2), -Fx)
            except np.linalg.LinAlgError:
                break
                
        if not np.all(np.isfinite(d)):
            break
            
        # Line search with Broyden update
        alpha = 1.0
        norm_current = _norm(Fx)
        found_step = False
        
        for _ls in range(max_ls_iter):
            if _elapsed() > deadline or calls >= 495:
                break
                
            x_new = x + alpha * d
            Fx_new = _safe_call(F, x_new)
            calls += 1
            
            if Fx_new is None:
                alpha *= 0.5
                continue
                
            norm_new = _norm(Fx_new)
            
            # Armijo condition (simplified): just require decrease
            if norm_new < norm_current:
                # Accept step
                s = x_new - x
                y = Fx_new - Fx
                
                # Broyden's Good Update
                # J = J + (y - J*s) * s^T / (s^T * s)
                Js = J @ s
                rank1 = (y - Js) @ (s / (s @ s)).T
                J = J + rank1
                
                x = x_new
                Fx = Fx_new
                found_step = True
                break
            else:
                alpha *= 0.5
                
        if not found_step:
            # If line search fails, we are stuck. 
            # Try a restart with a perturbed Jacobian or x?
            # Simple fallback: try a small random step?
            # Or break.
            break
            
        if _norm(Fx) <= 1e-12:
            break
            
    return x
