import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Homotopy path-following from origin to get a good initial point.
    2. Broyden's method with periodic FD Jacobian re-anchoring.
    3. Damped line search for robustness.
    4. Multiple diverse restarts if initial attempt fails.
    5. Final polish step with high-precision FD Jacobian if close.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    def time_ok():
        return (time.time() - start_time) < time_limit
    
    # Safe F evaluation with call counting
    call_count = [0]
    call_limit = 490  # Stay under 500
    
    def safe_eval_F(x):
        """Evaluate F with error handling and call counting."""
        try:
            result = np.array(F(x), dtype=float)
            call_count[0] += 1
            if result.shape != (2,) or not np.all(np.isfinite(result)):
                return None
            return result
        except Exception:
            call_count[0] += 1
            return None
    
    def compute_J_fd(x, Fx, h=1e-5):
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
    
    def compute_J_central(x, h=1e-6):
        """Compute Jacobian via central finite differences (4 calls)."""
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = safe_eval_F(x + e)
            Fx_minus = safe_eval_F(x - e)
            if Fx_plus is None or Fx_minus is None:
                return None
            J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
        return J
    
    def homotopy_from_origin(x0, max_iter=50, alpha_init=0.1):
        """
        Homotopy path-following from origin to x0.
        Solve F(x + t*(x0 - x)) = 0 for t going from 0 to 1.
        """
        # Start at origin
        x = np.zeros(2)
        t = 0.0
        alpha = alpha_init
        
        Fx = safe_eval_F(x)
        if Fx is None:
            return x0, None
        
        # If origin is already a root, great
        if np.linalg.norm(Fx) <= 1e-8:
            return x, Fx
        
        # Compute initial Jacobian at origin
        J = compute_J_fd(x, Fx, 1e-5)
        if J is None:
            J = compute_J_central(x, 1e-6)
            if J is None:
                return x0, None
        
        for it in range(max_iter):
            if not time_ok() or call_count[0] >= call_limit:
                break
            
            # Solve homotopy equation: F(x + t*d) = 0 where d = x0 - x
            # Use Newton-like step to increase t
            d = x0 - x
            x_t = x + t * d
            
            Fx_t = safe_eval_F(x_t)
            if Fx_t is None:
                break
            
            fx_norm = np.linalg.norm(Fx_t)
            if fx_norm <= 1e-8:
                return x_t, Fx_t
            
            # Compute Jacobian at x_t
            if it % 5 == 0:
                J = compute_J_fd(x_t, Fx_t, 1e-6)
                if J is None:
                    J = compute_J_central(x_t, 1e-6)
                    if J is None:
                        break
            
            # Solve for step in t
            # dF/dt = J @ d
            Jd = J @ d
            # Newton step: t_new = t - Fx_t / (Jd @ Fx_t)
            denom = float(np.dot(Jd, Fx_t))
            if abs(denom) < 1e-12:
                # Use damped update
                t_new = min(1.0, t + alpha)
            else:
                dt = -float(np.dot(Fx_t, Fx_t)) / denom
                # Limit step size
                dt = max(-0.5, min(0.5, dt))
                t_new = t + dt
            
            # Ensure t stays in [0, 1]
            t_new = max(0.0, min(1.0, t_new))
            
            x_new = x + t_new * d
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                break
            
            fx_new_norm = np.linalg.norm(Fx_new)
            if fx_new_norm <= 1e-8:
                return x_new, Fx_new
            
            # Broyden update for Jacobian
            s = x_new - x_t
            y = Fx_new - Fx_t
            s_dot_s = float(np.dot(s, s))
            if s_dot_s > 1e-16:
                J = J + np.outer(y - J @ s, s) / s_dot_s
            
            x = x_new
            t = t_new
            Fx = Fx_new
            
            # Check if we've reached t=1
            if t >= 0.999:
                break
            
            # Adaptive alpha
            if fx_new_norm > fx_norm:
                alpha *= 0.5
            else:
                alpha = min(1.0, alpha * 1.2)
        
        return x, Fx
    
    def broyden_hybrid(x_start, Fx_start=None, max_iter=35, reanchor_every=5):
        """
        Broyden's method with periodic re-anchoring.
        Returns (x, Fx, success).
        """
        x = np.array(x_start, dtype=float)
        
        # Initial evaluation
        if Fx_start is None:
            Fx = safe_eval_F(x)
        else:
            Fx = np.array(Fx_start, dtype=float)
        if Fx is None:
            return x, None, False
        if np.linalg.norm(Fx) <= 1e-8:
            return x, Fx, True
        
        # Initial Jacobian
        J = compute_J_fd(x, Fx, 1e-5)
        if J is None:
            J = compute_J_central(x, 1e-6)
            if J is None:
                return x, Fx, False
        
        # Check invertibility
        try:
            np.linalg.solve(J, np.ones(2))
        except np.linalg.LinAlgError:
            # Try different perturbation
            J = compute_J_fd(x, Fx, 1e-4)
            if J is None:
                return x, Fx, False
            try:
                np.linalg.solve(J, np.ones(2))
            except np.linalg.LinAlgError:
                return x, Fx, False
        
        best_x = x.copy()
        best_Fx = Fx.copy()
        best_norm = np.linalg.norm(Fx)
        
        for it in range(max_iter):
            if not time_ok() or call_count[0] >= call_limit:
                break
            
            # Periodic re-anchoring: recompute full Jacobian
            if it > 0 and it % reanchor_every == 0:
                J_new = compute_J_fd(x, Fx, 1e-6)
                if J_new is not None:
                    J = J_new
                else:
                    J_new = compute_J_central(x, 1e-6)
                    if J_new is not None:
                        J = J_new
            
            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Recompute Jacobian
                J = compute_J_fd(x, Fx, 1e-5)
                if J is None:
                    break
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    break
            
            # Damping
            step_norm = np.linalg.norm(dx)
            if step_norm > 30.0:
                dx = dx * (30.0 / step_norm)
            
            # Line search
            x_new = x + dx
            Fx_new = safe_eval_F(x_new)
            
            if Fx_new is None:
                for alpha in [0.5, 0.25, 0.1, 0.05]:
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
            
            # Broyden rank-1 update
            s = x_new - x
            y = Fx_new - Fx
            s_dot_s = np.dot(s, s)
            
            if s_dot_s > 1e-16:
                J = J + np.outer(y - J @ s, s) / s_dot_s
            
            x = x_new
            Fx = Fx_new
        
        return best_x, best_Fx, best_norm <= 1e-8
    
    def polish(x_start, Fx_start):
        """
        Final polish using Newton's method with high-precision FD Jacobian.
        Only called if we're close to a root.
        """
        x = np.array(x_start, dtype=float)
        Fx = np.array(Fx_start, dtype=float)
        
        fx_norm = np.linalg.norm(Fx)
        if fx_norm > 1e-4:
            return x, Fx, False
        
        for it in range(10):
            if not time_ok() or call_count[0] >= call_limit:
                break
            
            # Central difference Jacobian for accuracy
            J = compute_J_central(x, 1e-7)
            if J is None:
                break
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            
            x_new = x + dx
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                break
            
            fx_new_norm = np.linalg.norm(Fx_new)
            if fx_new_norm <= 1e-8:
                return x_new, Fx_new, True
            
            # Check for improvement
            if fx_new_norm >= fx_norm:
                break
            
            x = x_new
            Fx = Fx_new
        
        return x, Fx, np.linalg.norm(Fx) <= 1e-8
    
    # Main algorithm
    x0 = np.array(x0, dtype=float)
    
    # First, try homotopy from origin to get a better starting point
    if time_ok() and call_count[0] < call_limit:
        x_hom, Fx_hom = homotopy_from_origin(x0)
        if Fx_hom is not None:
            x_start = x_hom
            Fx_start = Fx_hom
        else:
            x_start = x0
            Fx_start = None
    else:
        x_start = x0
        Fx_start = None
    
    # Run Broyden hybrid
    x_result, Fx_result, success = broyden_hybrid(x_start, Fx_start)
    
    if success:
        return x_result
    
    # If Broyden didn't converge, try a few more with fresh Jacobian
    if time_ok() and call_count[0] < call_limit:
        # Try from original x0
        x_try, Fx_try, success = broyden_hybrid(x0, None, max_iter=20)
        if success:
            return x_try
        
        # Try from homotopy result if different
        if not np.allclose(x_start, x0):
            x_try, Fx_try, success = broyden_hybrid(x_start, None, max_iter=20)
            if success:
                return x_try
    
    # Try polish if we're close
    if Fx_result is not None and np.linalg.norm(Fx_result) < 1e-4:
        x_polished, Fx_polished, success = polish(x_result, Fx_result)
        if success:
            return x_polished
    
    # Return best result found
    return x_result if Fx_result is not None else x0
