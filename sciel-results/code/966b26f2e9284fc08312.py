"""Root finder for 3 fixed 2-D systems: Secant-Step Hybrid Broyden with re-anchoring and polish."""
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Secant-Step Hybrid: initialize Jacobian via 2-point secant (2 calls),
       then alternate Broyden rank-1 updates with periodic full FD recomputation
       every k iterations to prevent quasi-Newton drift.
    2. If stuck, try restarts from scaled/random versions of x0.
    3. Final high-precision polish pass with tighter tolerance.
    
    Budget:
    - Time limit: ~118s internal (evaluator kills at 150s).
    - Call limit: < 500 calls per system.
    """
    start_time = time.time()
    time_limit = 118.0
    
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
    
    def compute_J_secant(x, Fx):
        """
        Two-point secant Jacobian estimate.
        Uses x and x + h*e (where e is a direction) to build J in 2 calls total
        (one for Fx already done, one for the perturbed point).
        Actually we need 2 additional calls for a 2x2 Jacobian via secant:
        J[:,0] ~ (F(x + h*e0) - F(x)) / h  -> 1 call
        J[:,1] ~ (F(x + h*e1) - F(x)) / h  -> 1 call
        That's 2 extra calls. Alternatively, use a single secant direction:
        Pick d = normalized Fx (or random if Fx ~ 0), evaluate F(x + h*d),
        then J ~ (F(x+h*d) - F(x)) * d^T / (d^T d)  -- rank-1, not full.
        
        Better: use 2 calls with two different perturbations to get a 2x2.
        We'll use forward differences in e0 and e1: 2 calls.
        """
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = safe_eval_F(x + e)
            if Fx_e is None:
                return None
            J[:, j] = (Fx_e - Fx) / h
        return J
    
    def compute_J_central(x, Fx):
        """
        Central-difference Jacobian: 4 calls (2 per column).
        More accurate for polish phase.
        """
        h = 1e-8
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_p = safe_eval_F(x + e)
            Fx_m = safe_eval_F(x - e)
            if Fx_p is None or Fx_m is None:
                return None
            J[:, j] = (Fx_p - Fx_m) / (2 * h)
        return J

    def compute_J_forward(x, Fx):
        """
        Forward-difference Jacobian: 2 calls.
        Adaptive step size based on norm of Fx for better conditioning.
        """
        norm_Fx = np.linalg.norm(Fx)
        # Scale step size: smaller when close to root for better accuracy
        h = max(1e-8, min(1e-4, 1e-5 * max(1.0, norm_Fx ** 0.5)))
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_p = safe_eval_F(x + e)
            if Fx_p is None:
                return None
            J[:, j] = (Fx_p - Fx) / h
        return J
    
    def broyden_hybrid(x_start, max_iter=40, reanchor_every=4):
        """
        Secant-Step Hybrid Broyden's method with adaptive damping.
        - Initial J via forward differences (2 calls).
        - Rank-1 Broyden updates each iteration.
        - Every reanchor_every iterations, recompute J via forward differences (2 calls)
          to correct accumulated error.
        - Adaptive step damping based on convergence rate.
        Returns (x, Fx, success).
        """
        x = np.array(x_start, dtype=float)
        
        # Initial evaluation
        Fx = safe_eval_F(x)
        if Fx is None:
            return x, None, False
        if np.linalg.norm(Fx) <= 1e-8:
            return x, Fx, True
        
        # Initial Jacobian via forward differences (2 calls)
        J = compute_J_secant(x, Fx)
        if J is None:
            return x, Fx, False
        
        # Check invertibility
        if abs(np.linalg.det(J)) < 1e-12:
            # Try larger h
            h_backup = 1e-5
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h_backup
                Fx_e = safe_eval_F(x + e)
                if Fx_e is None:
                    return x, Fx, False
                J[:, j] = (Fx_e - Fx) / h_backup
            if abs(np.linalg.det(J)) < 1e-12:
                return x, Fx, False
        
        best_x = x.copy()
        best_Fx = Fx.copy()
        best_norm = np.linalg.norm(Fx)
        
        # Adaptive damping factor
        damp_factor = 1.0
        
        for it in range(max_iter):
            if not time_ok() or call_count[0] >= 490:
                break
            
            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Re-anchor before giving up
                if it < max_iter - 1:
                    J_new = compute_J_secant(x, Fx)
                    if J_new is not None and abs(np.linalg.det(J_new)) > 1e-12:
                        J = J_new
                        continue
                break
            
            # Adaptive damping: reduce step if previous steps were large
            step_norm = np.linalg.norm(dx)
            max_step = 2.0 * damp_factor
            if step_norm > max_step:
                dx = dx * (max_step / step_norm)
            
            x_new = x + dx
            
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                # Halve the step
                x_new = x + 0.5 * dx
                Fx_new = safe_eval_F(x_new)
                if Fx_new is None:
                    x_new = x + 0.25 * dx
                    Fx_new = safe_eval_F(x_new)
                    if Fx_new is None:
                        break
            
            Fx_new_norm = np.linalg.norm(Fx_new)
            
            # Track best and adapt damping
            if Fx_new_norm < best_norm:
                best_x = x_new.copy()
                best_Fx = Fx_new.copy()
                best_norm = Fx_new_norm
                # If making good progress, allow larger steps
                if Fx_new_norm < 0.5 * np.linalg.norm(Fx):
                    damp_factor = min(1.0, damp_factor * 1.2)
            else:
                # If not improving, reduce step size
                damp_factor = max(0.3, damp_factor * 0.7)
            
            if Fx_new_norm <= 1e-8:
                return x_new, Fx_new, True
            
            # Update Jacobian
            s = x_new - x
            y = Fx_new - Fx
            s_dot_s = np.dot(s, s)
            
            if (it + 1) % reanchor_every == 0:
                # Re-anchor: recompute J from scratch (2 calls)
                J_new = compute_J_secant(x_new, Fx_new)
                if J_new is not None and abs(np.linalg.det(J_new)) > 1e-12:
                    J = J_new
                elif s_dot_s > 1e-16:
                    # Fall back to Broyden update
                    J = J + np.outer(y - J @ s, s) / s_dot_s
            else:
                # Standard Broyden rank-1 update
                if s_dot_s > 1e-16:
                    J = J + np.outer(y - J @ s, s) / s_dot_s
            
            x = x_new
            Fx = Fx_new
        
        return best_x, best_Fx, best_norm <= 1e-8
    
    def polish(x_start):
        """
        High-precision polish: use secant-based Newton with minimal calls.
        When close to root, we can estimate Jacobian from previous step
        to avoid recomputing differences.
        """
        x = np.array(x_start, dtype=float)
        Fx = safe_eval_F(x)
        if Fx is None:
            return x
        norm = np.linalg.norm(Fx)
        if norm <= 1e-8:
            return x
        
        # Only polish if reasonably close
        if norm > 1e-3:
            return x
        
        # Compute initial Jacobian via forward differences (2 calls)
        J = compute_J_forward(x, Fx)
        if J is None:
            return x
        if abs(np.linalg.det(J)) < 1e-14:
            return x
        
        prev_x = x.copy()
        prev_Fx = Fx.copy()
        
        for it in range(8):
            if not time_ok() or call_count[0] >= 498:
                break
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            
            step_norm = np.linalg.norm(dx)
            if step_norm > 0.1:
                dx = dx * (0.1 / step_norm)
            
            x_new = x + dx
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                break
            
            Fx_new_norm = np.linalg.norm(Fx_new)
            
            if Fx_new_norm < norm:
                # Accept step and update Jacobian via secant
                s = x_new - prev_x
                y = Fx_new - prev_Fx
                s_dot_s = np.dot(s, s)
                if s_dot_s > 1e-18:
                    J = J + np.outer(y - J @ s, s) / s_dot_s
                
                prev_x = x.copy()
                prev_Fx = Fx.copy()
                x = x_new
                Fx = Fx_new
                norm = Fx_new_norm
                if norm <= 1e-8:
                    return x
            else:
                # Try half step
                x_half = x + 0.5 * dx
                Fx_half = safe_eval_F(x_half)
                if Fx_half is not None:
                    Fx_half_norm = np.linalg.norm(Fx_half)
                    if Fx_half_norm < norm:
                        s = x_half - prev_x
                        y = Fx_half - prev_Fx
                        s_dot_s = np.dot(s, s)
                        if s_dot_s > 1e-18:
                            J = J + np.outer(y - J @ s, s) / s_dot_s
                        
                        prev_x = x.copy()
                        prev_Fx = Fx.copy()
                        x = x_half
                        Fx = Fx_half
                        norm = Fx_half_norm
                        if norm <= 1e-8:
                            return x
                break
        
        return x
    
    # First attempt: Secant-Step Hybrid Broyden from x0
    x0_arr = np.array(x0, dtype=float)
    x, Fx, success = broyden_hybrid(x0_arr, max_iter=35, reanchor_every=4)
    if success:
        return x
    
    # If not successful but close, try polish
    if Fx is not None and np.linalg.norm(Fx) <= 1e-3:
        x_polished = polish(x)
        Fx_check = safe_eval_F(x_polished)
        if Fx_check is not None and np.linalg.norm(Fx_check) <= 1e-8:
            return x_polished
    
    # If first attempt failed, try restarts
    restarts = []
    
    # Scaled versions of x0 (fewer, more targeted)
    for scale in [0.5, 2.0, 0.1, -1.0]:
        restarts.append(scale * x0_arr)
    
    # Random perturbations (fewer)
    rng = np.random.RandomState(42)
    x0_norm = np.linalg.norm(x0_arr)
    if x0_norm < 1e-12:
        x0_norm = 1.0
    for _ in range(3):
        restarts.append(x0_arr + rng.randn(2) * x0_norm * 0.3)
    
    # Origin
    restarts.append(np.zeros(2))
    
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx) if Fx is not None else np.inf
    
    for r in restarts:
        if not time_ok() or call_count[0] >= 465:
            break
        
        x_r, Fx_r, success_r = broyden_hybrid(r, max_iter=15, reanchor_every=4)
        if success_r:
            return x_r
        
        if Fx_r is not None:
            norm_r = np.linalg.norm(Fx_r)
            if norm_r < best_norm:
                best_norm = norm_r
                best_x = x_r.copy()
    
    # Final polish on best candidate
    if best_norm <= 1e-3:
        x_polished = polish(best_x)
        Fx_check = safe_eval_F(x_polished)
        if Fx_check is not None and np.linalg.norm(Fx_check) <= 1e-8:
            return x_polished
        if Fx_check is not None:
            best_x = x_polished
    
    return best_x
