# EVOLVE-BLOCK-START
import numpy as np
import time

def _safe_norm(F, x):
    """Safely compute norm of F(x), returning inf on failure."""
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if not np.all(np.isfinite(Fx)):
            return np.inf
        return np.linalg.norm(Fx)
    except Exception:
        return np.inf

def _newton_step(F, x, J=None):
    """
    Perform one Newton step.
    If J is not provided, estimate it via finite differences.
    Returns (x_new, J_new, success).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    
    try:
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            return x, J, False
        
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx < 1e-12:
            return x, J, True
            
        if J is None:
            # Finite difference Jacobian
            h = 1e-7
            J = np.zeros((n, n))
            for j in range(n):
                x_p = x.copy()
                x_p[j] += h
                Fx_p = F(x_p)
                if not np.all(np.isfinite(Fx_p)):
                    return x, J, False
                J[:, j] = (Fx_p - Fx) / h
        else:
            J = np.asarray(J, dtype=float)
            
        # Solve J dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                return x, J, False
        
        # Damping if step is too large
        step_norm = np.linalg.norm(dx)
        if step_norm > 1.0:
            dx = dx * (1.0 / step_norm)
            
        x_new = x + dx
        if not np.all(np.isfinite(x_new)):
            return x, J, False
            
        return x_new, J, True
        
    except Exception:
        return x, J, False

def _track_root_homotopy(F, x0, max_steps=200, dt_initial=0.1, dt_min=1e-3):
    """
    Track the root of F(x)=0 by homotopy from a known root.
    We define G(x) = x - x_target. Root of G is x_target.
    H(x, t) = (1-t)F(x) + t G(x).
    At t=1, H(x, 1) = G(x), root is x_target.
    At t=0, H(x, 0) = F(x), root is the desired root.
    
    We start at x_target and decrease t to 0.
    """
    x = np.array(x0, dtype=float)
    
    # Define the target for the homotopy. 
    # Using x0 as the "known" root is risky if F(x0) is far from 0.
    # Better: Use a generic linear system G(x) = x. Root is 0.
    # But if the root of F is far from 0, we might lose it.
    # Alternative: Use G(x) = x - x0. Root is x0.
    # Let's try G(x) = x. This assumes the root is in a region connected to 0.
    # If that fails, we can try G(x) = x - x0.
    
    # Strategy: Try homotopy from 0 first. If it fails or goes out of bounds, 
    # the caller can try from x0.
    
    x_current = np.zeros(2, dtype=float)
    t = 1.0
    dt = dt_initial
    
    # We need to track the root of H(x, t) = 0.
    # H(x, t) = (1-t)F(x) + t*x
    
    # Initial check at t=1: H(x, 1) = x. Root is 0.
    # We start at x=0.
    
    for step in range(max_steps):
        if t <= 0:
            break
            
        # Predictor step
        t_new = max(0.0, t - dt)
        
        # We want to find x_new such that H(x_new, t_new) = 0.
        # Approximate x_new using Newton on the homotopy.
        
        # Define homotopy function for Newton
        def H_func(x_val):
            Fx = F(x_val)
            if not np.all(np.isfinite(Fx)):
                return None
            return (1.0 - t_new) * Fx + t_new * x_val
        
        # Jacobian of H: (1-t)J_F + t*I
        # We don't know J_F exactly, but we can estimate it or use previous J.
        # For simplicity and robustness, we will use a few Newton iterations 
        # with FD Jacobian for the homotopy function.
        
        x_pred = x_current
        J_H = None
        
        converged = False
        for _ in range(10):
            try:
                H_val = H_func(x_pred)
                if H_val is None:
                    break
                norm_H = np.linalg.norm(H_val)
                if norm_H < 1e-10:
                    converged = True
                    break
                
                # Estimate J_H
                if J_H is None:
                    h = 1e-7
                    J_H = np.zeros((2, 2))
                    for j in range(2):
                        x_p = x_pred.copy()
                        x_p[j] += h
                        H_p = H_func(x_p)
                        if H_p is None:
                            break
                        J_H[:, j] = (H_p - H_val) / h
                    if not np.all(np.isfinite(J_H)):
                        break
                else:
                    # Reuse J_H? It changes with t. Better to recompute or update.
                    # For now, recompute every Newton step to be safe, though expensive.
                    h = 1e-7
                    J_H_new = np.zeros((2, 2))
                    for j in range(2):
                        x_p = x_pred.copy()
                        x_p[j] += h
                        H_p = H_func(x_p)
                        if H_p is None:
                            break
                        J_H_new[:, j] = (H_p - H_val) / h
                    if np.all(np.isfinite(J_H_new)):
                        J_H = J_H_new
                    else:
                        break
                        
                try:
                    dx = np.linalg.solve(J_H, -H_val)
                except np.linalg.LinAlgError:
                    break
                    
                # Damping
                step_norm = np.linalg.norm(dx)
                if step_norm > 1.0:
                    dx = dx / step_norm
                    
                x_pred = x_pred + dx
                if not np.all(np.isfinite(x_pred)):
                    break
                    
            except Exception:
                break
                
        if converged and np.all(np.isfinite(x_pred)):
            x_current = x_pred
            t = t_new
            # Adaptive step size: if we converged easily, increase dt? 
            # If we struggled, decrease dt.
            # Simple heuristic: keep dt constant or reduce if t is small.
            if t < 0.1:
                dt = max(dt_min, dt * 0.5)
        else:
            # Failed to track. Try smaller step?
            dt = max(dt_min, dt * 0.5)
            if dt < dt_min * 0.5:
                break
                
    return x_current

def find_root_system(F, x0):
    """
    Finds a root x such that ||F(x)|| <= 1e-8.
    Strategy:
    1. Check x0.
    2. Try homotopy tracking from 0.
    3. Try homotopy tracking from x0 (using G(x) = x - x0).
    4. Fallback to local Newton/Broyden from best candidate.
    """
    start_time = time.time()
    DEADLINE = 110.0
    
    x0 = np.array(x0, dtype=float)
    
    # 1. Check x0
    norm0 = _safe_norm(F, x0)
    if norm0 <= 1e-8:
        return x0
        
    best_x = x0
    best_norm = norm0
    
    # 2. Homotopy from 0
    # This assumes the root is connected to 0 in a way that H(x,t) doesn't vanish or become singular.
    if time.time() - start_time < DEADLINE * 0.5:
        x_h1 = _track_root_homotopy(F, x0, max_steps=100, dt_initial=0.05, dt_min=1e-3)
        norm_h1 = _safe_norm(F, x_h1)
        if norm_h1 < best_norm:
            best_norm = norm_h1
            best_x = x_h1
        if norm_h1 <= 1e-8:
            return x_h1
            
    # 3. Homotopy from x0 (Target G(x) = x - x0)
    # We modify the tracker to use x0 as the base.
    if time.time() - start_time < DEADLINE * 0.7:
        # We need a variant of _track_root_homotopy that uses x0.
        # Let's implement a quick inline version or modify the function to accept a base point.
        # For now, let's just try Newton from x0 with a few restarts if homotopy from 0 failed.
        
        # Actually, let's try a "radial" Newton approach.
        # If homotopy from 0 failed, maybe the root is far.
        
        # Try Broyden from x0
        x_b, norm_b, _ = _broyden_local(F, x0)
        if norm_b < best_norm:
            best_norm = norm_b
            best_x = x_b
        if norm_b <= 1e-8:
            return x_b
            
    # 4. Multi-start Broyden from a few deterministic points
    # This is a fallback to ensure we have some global coverage.
    if time.time() - start_time < DEADLINE * 0.9:
        offsets = [
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
            np.array([1.0, 1.0]),
            np.array([-1.0, -1.0]),
            np.array([10.0, 0.0]),
            np.array([-10.0, 0.0]),
        ]
        
        for off in offsets:
            if time.time() - start_time > DEADLINE * 0.95:
                break
            x_try = x0 + off
            x_b, norm_b, _ = _broyden_local(F, x_try)
            if norm_b < best_norm:
                best_norm = norm_b
                best_x = x_b
            if norm_b <= 1e-8:
                return x_b
                
    return best_x

def _broyden_local(F, x0, max_iter=30):
    """
    Local Broyden's method starting from x0.
    """
    x = np.array(x0, dtype=float)
    n = 2
    
    try:
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            return x, np.inf, False
            
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-8:
            return x, norm_Fx, True
            
        # Initial Jacobian
        h = 1e-7
        J = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            Fx_p = F(x + e)
            if not np.all(np.isfinite(Fx_p)):
                return x, np.inf, False
            J[:, j] = (Fx_p - Fx) / h
            
        for _ in range(max_iter):
            if norm_Fx <= 1e-12:
                return x, norm_Fx, True
                
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except Exception:
                    return x, norm_Fx, False
                    
            # Damping
            step_norm = np.linalg.norm(dx)
            if step_norm > 1.0:
                dx = dx / step_norm
                
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                return x, norm_Fx, False
                
            Fx_new = F(x_new)
            if not np.all(np.isfinite(Fx_new)):
                return x, norm_Fx, False
                
            norm_Fx_new = np.linalg.norm(Fx_new)
            
            # Broyden update
            y = Fx_new - Fx
            Jdx = J @ dx
            delta = y - Jdx
            dx_norm_sq = np.dot(dx, dx)
            if dx_norm_sq > 1e-30:
                J = J + np.outer(delta, dx) / dx_norm_sq
                
            x = x_new
            Fx = Fx_new
            norm_Fx = norm_Fx_new
            
        return x, norm_Fx, False
        
    except Exception:
        return x, np.inf, False
# EVOLVE-BLOCK-END
