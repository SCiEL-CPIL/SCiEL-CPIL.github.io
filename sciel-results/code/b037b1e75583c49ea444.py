"""Root finder for 3 fixed 2-D systems: Broyden's method with safeguards."""
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Start with x0 and use Broyden's method (good initial Jacobian estimate).
    2. If stuck, try a few random restarts from scaled versions of x0.
    3. Use damping if step is too large.
    """
    start_time = time.time()
    time_limit = 115.0
    
    def time_ok():
        return (time.time() - start_time) < time_limit
    
    # Safe F evaluation that counts calls and handles errors
    call_count = [0]
    
    def safe_eval_F(x):
        """Evaluate F with error handling."""
        try:
            result = np.array(F(x), dtype=float)
            call_count[0] += 1
            if result.shape != (2,) or not np.all(np.isfinite(result)):
                return None
            return result
        except Exception:
            call_count[0] += 1
            return None
    
    def compute_J_fd(x, Fx, h=1e-6):
        """Compute Jacobian via forward finite differences (2 calls)."""
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = safe_eval_F(x + e)
            if Fx_e is None:
                return None
            J[:, j] = (Fx_e - Fx) / h
        return J
    
    def broyden(x_start, max_iter=40):
        """
        Broyden's method with safeguards.
        Returns (x, Fx, success) where success is True if ||Fx|| <= 1e-8.
        """
        x = np.array(x_start, dtype=float)
        
        # Initial evaluation
        Fx = safe_eval_F(x)
        if Fx is None:
            return x, None, False
        if np.linalg.norm(Fx) <= 1e-8:
            return x, Fx, True
        
        # Initial Jacobian estimate using forward differences (2 calls)
        h = 1e-6
        J = compute_J_fd(x, Fx, h)
        if J is None:
            return x, Fx, False
        
        # Check if J is invertible
        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-12:
            # Try a different perturbation
            h = 1e-4
            J = compute_J_fd(x, Fx, h)
            if J is None:
                return x, Fx, False
            det_J = np.linalg.det(J)
            if abs(det_J) < 1e-12:
                return x, Fx, False
        
        # Broyden's iterations
        best_x = x.copy()
        best_Fx = Fx.copy()
        best_norm = np.linalg.norm(Fx)
        
        for it in range(max_iter):
            if not time_ok() or call_count[0] >= 470:
                break
            
            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Try to recover by recomputing J
                Fx_new_J = safe_eval_F(x)
                if Fx_new_J is None:
                    break
                J = compute_J_fd(x, Fx_new_J, 1e-5)
                if J is None:
                    break
                try:
                    dx = np.linalg.solve(J, -Fx_new_J)
                except np.linalg.LinAlgError:
                    break
                Fx = Fx_new_J
                if np.linalg.norm(Fx) < best_norm:
                    best_x = x.copy()
                    best_Fx = Fx.copy()
                    best_norm = np.linalg.norm(Fx)
                continue
            
            # Damping: if step is too large, reduce it
            step_norm = np.linalg.norm(dx)
            if step_norm > 50.0:
                dx = dx * (50.0 / step_norm)
            
            # Line search: try full step, then halve
            x_new = x + dx
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                # Try smaller steps
                for alpha in [0.5, 0.25, 0.1]:
                    x_new = x + alpha * dx
                    Fx_new = safe_eval_F(x_new)
                    if Fx_new is not None:
                        break
                if Fx_new is None:
                    break
            
            Fx_new_norm = np.linalg.norm(Fx_new)
            
            # Track best
            if Fx_new_norm < best_norm:
                best_x = x_new.copy()
                best_Fx = Fx_new.copy()
                best_norm = Fx_new_norm
            
            if Fx_new_norm <= 1e-8:
                return x_new, Fx_new, True
            
            # Update Jacobian (Broyden's rank-1 update)
            s = x_new - x
            y = Fx_new - Fx
            s_dot_s = np.dot(s, s)
            
            if s_dot_s > 1e-16:
                J = J + np.outer(y - J @ s, s) / s_dot_s
            
            x = x_new
            Fx = Fx_new
        
        return best_x, best_Fx, best_norm <= 1e-8
    
    # First attempt: Broyden from x0
    x0_arr = np.array(x0, dtype=float)
    x, Fx, success = broyden(x0_arr, max_iter=40)
    if success:
        return x
    
    # If first attempt failed, try restarts
    restarts = []
    
    # Scaled versions of x0
    for scale in [0.5, 2.0, 0.1, 5.0, -1.0, -0.5]:
        restarts.append(scale * x0_arr)
    
    # Random perturbations
    rng = np.random.RandomState(42)
    for _ in range(8):
        restarts.append(x0_arr + rng.randn(2) * max(np.linalg.norm(x0_arr), 1.0) * 0.5)
    
    # Origin
    restarts.append(np.zeros(2))
    
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx) if Fx is not None else np.inf
    
    for r in restarts:
        if not time_ok() or call_count[0] >= 440:
            break
        
        x_r, Fx_r, success_r = broyden(r, max_iter=30)
        if success_r:
            return x_r
        
        if Fx_r is not None:
            norm_r = np.linalg.norm(Fx_r)
            if norm_r < best_norm:
                best_norm = norm_r
                best_x = x_r.copy()
    
    # Final check on best_x
    if Fx is not None and np.linalg.norm(Fx) <= 1e-8:
        return best_x
    
    return best_x
