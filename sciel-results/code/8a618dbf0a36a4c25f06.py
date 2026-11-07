import numpy as np
import time

_DEADLINE_FRAC = 0.85
_CALL_LIMIT = 480

def _elapsed_ok(start, limit_frac=0.85, wall=120.0):
    return (time.time() - start) < limit_frac * wall

def _finite2(x):
    return (isinstance(x, np.ndarray) and x.shape == (2,) and np.all(np.isfinite(x)))

def _norm2(v):
    return float(np.linalg.norm(v))

def find_root_system(F, x0):
    start = time.time()
    x = np.array(x0, dtype=float)
    
    # Initial evaluation
    try:
        Fx = np.asarray(F(x), dtype=float)
    except Exception:
        return x
    
    if not _finite2(Fx):
        return x
        
    best_x = x.copy()
    best_norm = _norm2(Fx)
    calls = 1

    if best_norm <= 1e-8:
        return best_x

    # Initial Jacobian estimate using Finite Differences
    # We need 2 more calls
    step = 1e-5
    J = np.zeros((2, 2))
    for j in range(2):
        if calls >= _CALL_LIMIT or not _elapsed_ok(start):
            break
        e = np.zeros(2)
        e[j] = step
        try:
            Fx_p = np.asarray(F(x + e), dtype=float)
            calls += 1
            if _finite2(Fx_p):
                J[:, j] = (Fx_p - Fx) / step
            else:
                # Fallback if FD fails
                J[:, j] = 0 
        except Exception:
            J[:, j] = 0
            
    if calls >= _CALL_LIMIT or not _elapsed_ok(start):
        return best_x

    # Broyden's Method Loop
    # We use Good-Broyden
    # J_{k+1} = J_k + ( (F_{k+1} - F_k) - J_k (x_{k+1} - x_k) ) (x_{k+1} - x_k)^T / ||x_{k+1} - x_k||^2
    
    for it in range(30): # Max 30 iterations
        if calls >= _CALL_LIMIT - 1 or not _elapsed_ok(start):
            break
            
        # Solve J dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If J is singular, perturb it or restart with FD
            # For simplicity, add small diagonal
            try:
                dx = np.linalg.solve(J + 1e-8*np.eye(2), -Fx)
            except:
                break
                
        if not np.all(np.isfinite(dx)):
            break
            
        # Try the step
        x_new = x + dx
        Fx_new = None
        used_in_step = 0
        
        # Simple damping if step is too large or norm increases too much
        # We will try the full step first.
        try:
            Fx_new = np.asarray(F(x_new), dtype=float)
            calls += 1
            used_in_step = 1
        except Exception:
            break
            
        if not _finite2(Fx_new):
            # Try half step
            x_new = x + 0.5 * dx
            try:
                Fx_new = np.asarray(F(x_new), dtype=float)
                calls += 1
                used_in_step = 2
            except Exception:
                break
                
            if not _finite2(Fx_new):
                # Try quarter step
                x_new = x + 0.25 * dx
                try:
                    Fx_new = np.asarray(F(x_new), dtype=float)
                    calls += 1
                    used_in_step = 3
                except Exception:
                    break
                    
                if not _finite2(Fx_new):
                    continue # Reject step

        norm_new = _norm2(Fx_new)
        
        if norm_new < best_norm:
            best_x = x_new.copy()
            best_norm = norm_new
            
            # Check convergence
            if best_norm <= 1e-8:
                return best_x
                
            # Update Jacobian
            delta_x = x_new - x
            delta_F = Fx_new - Fx
            norm_dx_sq = np.dot(delta_x, delta_x)
            if norm_dx_sq > 1e-16:
                # Good Broyden update
                err = delta_F - J @ delta_x
                J = J + np.outer(err, delta_x) / norm_dx_sq
            else:
                # If step was tiny, don't update J or use FD?
                pass
                
            x = x_new
            Fx = Fx_new
        else:
            # Step rejected (norm increased)
            # We might try to recover by resetting J to FD or just damping more?
            # To save calls, we just reject and try a smaller step?
            # But we already used calls for the failed steps.
            # Let's try a smaller step in the next iteration by scaling dx?
            # Or just break if we are stuck?
            # A robust approach: if we reject, we can try to update J with the failed point? No.
            # Let's just try a damped step immediately?
            # Actually, the code above already tried 1.0, 0.5, 0.25.
            # If all failed to improve, we are stuck.
            # Let's try one more time with a very small step?
            # Or just break to avoid wasting calls.
            break
            
    return best_x
