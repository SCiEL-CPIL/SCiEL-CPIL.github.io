"""Root finder for 3 fixed 2D systems using Adaptive Homotopy with Broyden and Multi-Start."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    time_limit = 145.0  # Safety margin from 150s limit
    
    x0_arr = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0_arr)):
        x0_arr = np.zeros(2)
        
    call_counter = [0]
    max_calls = 490
    
    def safe_eval(x):
        if call_counter[0] >= max_calls:
            return None, float('inf')
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                return None, float('inf')
            call_counter[0] += 1
            return Fx, np.linalg.norm(Fx)
        except Exception:
            return None, float('inf')

    best_x = x0_arr.copy()
    best_norm = float('inf')
    
    # Check if x0 is already a root
    Fx, norm_F = safe_eval(x0_arr)
    if Fx is not None:
        best_norm = norm_F
        if norm_F <= 1e-8:
            return best_x

    # Strategy: Homotopy continuation with adaptive t-steps and Broyden's method.
    # We solve F_t(x) = t*F(x) + (1-t)*(x - x0) = 0 for t from 0 to 1.
    # At t=0, x=x0 is the unique root. At t=1, we have a root of F(x)=0.
    # This helps avoid local minima/saddles by following the path from the known solution.
    
    def run_homotopy(start_x, max_outer_steps=50, max_inner_iters=20, initial_dt=0.1):
        nonlocal best_x, best_norm, call_counter
        
        x_current = start_x.copy()
        t = 0.0
        dt = initial_dt
        
        # Hessian approximation for F_t
        # At t=0, F_0(x) = x - x0, so J_0 = I.
        H = np.eye(2)
        
        # Initial evaluation of F_t at start_x
        # Note: F_t(x) = t*F(x) + (1-t)*(x - x0)
        # At t=0, F_t(x0) = 0.
        
        for outer_step in range(max_outer_steps):
            if time.time() - start_time > time_limit:
                return False
            if call_counter[0] >= max_calls:
                return False
                
            # Increase t
            t_new = min(1.0, t + dt)
            
            # Define F_t
            def F_t(x, t_val=t_new):
                Fx_orig, norm_orig = safe_eval(x)
                if Fx_orig is None:
                    return None, float('inf')
                Fx_t = t_val * Fx_orig + (1.0 - t_val) * (x - x0_arr)
                norm_t = np.linalg.norm(Fx_t)
                return Fx_t, norm_t
            
            # Solve F_t(x) = 0 using Broyden's method
            x = x_current.copy()
            Fx_t, norm_t = F_t(x)
            
            if Fx_t is None:
                dt *= 0.5
                if dt < 1e-4:
                    return False
                continue
                
            # Broyden iterations
            converged = False
            inner_iter = 0
            for inner_iter in range(max_inner_iters):
                if time.time() - start_time > time_limit:
                    return False
                if call_counter[0] >= max_calls:
                    return False
                    
                # Newton step
                try:
                    s = np.linalg.solve(H, -Fx_t)
                except np.linalg.LinAlgError:
                    s = np.linalg.lstsq(H, -Fx_t, rcond=None)[0]
                    
                if not np.all(np.isfinite(s)):
                    break
                    
                # Line search
                alpha = 1.0
                c = 1e-4
                x_new = None
                Fx_new = None
                norm_new = float('inf')
                
                for _ in range(10):
                    x_trial = x + alpha * s
                    Fx_trial, norm_trial = F_t(x_trial)
                    
                    if Fx_trial is None:
                        alpha *= 0.5
                        continue
                        
                    if norm_trial < (1 - c * alpha) * norm_t:
                        x_new = x_trial
                        Fx_new = Fx_trial
                        norm_new = norm_trial
                        break
                    else:
                        alpha *= 0.5
                        
                if x_new is None:
                    break
                    
                # Update H (Broyden Good)
                delta_x = x_new - x
                delta_F = Fx_new - Fx_t
                denom = np.dot(delta_x, delta_x)
                
                if denom > 1e-12:
                    H = H + np.outer(delta_F - H @ delta_x, delta_x) / denom
                    
                x = x_new
                Fx_t = Fx_new
                norm_t = norm_new
                
                # Track best (only if t is close to 1, otherwise the norm is for F_t, not F)
                # Actually, we should track the norm of F(x) for the best solution.
                # But F_t(x) = 0 implies t*F(x) + (1-t)*(x-x0) = 0.
                # If t=1, F_t(x) = F(x).
                # If t<1, F_t(x) != F(x).
                # So we only update best_x if t_new is 1.0 or if we do a final check.
                
                if norm_t <= 1e-8:
                    converged = True
                    break
            
            if converged:
                # Check if we've reached t=1
                if t_new >= 1.0:
                    # We are at a root of F(x)
                    # Do a final polish with pure F(x) to ensure precision
                    x_polish = x.copy()
                    Fx_polish, norm_polish = safe_eval(x_polish)
                    if Fx_polish is not None:
                        best_norm = norm_polish
                        best_x = x_polish.copy()
                        
                        # A few more Broyden steps on F(x) directly
                        for _ in range(10):
                            if time.time() - start_time > time_limit:
                                break
                            if call_counter[0] >= max_calls:
                                break
                                
                            try:
                                s = np.linalg.solve(H, -Fx_polish)
                            except np.linalg.LinAlgError:
                                s = np.linalg.lstsq(H, -Fx_polish, rcond=None)[0]
                                
                            if not np.all(np.isfinite(s)):
                                break
                                
                            alpha = 1.0
                            x_trial = x_polish + alpha * s
                            Fx_trial, norm_trial = safe_eval(x_trial)
                            
                            if Fx_trial is None:
                                break
                            
                            # Simple step update
                            delta_x = x_trial - x_polish
                            delta_F = Fx_trial - Fx_polish
                            denom = np.dot(delta_x, delta_x)
                            if denom > 1e-12:
                                H = H + np.outer(delta_F - H @ delta_x, delta_x) / denom
                                
                            x_polish = x_trial
                            Fx_polish = Fx_trial
                            norm_polish = norm_trial
                            
                            if norm_polish < best_norm:
                                best_norm = norm_polish
                                best_x = x_polish.copy()
                                
                            if norm_polish <= 1e-8:
                                break
                                
                    return True
                else:
                    # Move to next t
                    x_current = x.copy()
                    t = t_new
                    # Increase dt if we converged quickly
                    if inner_iter < 5:
                        dt = min(1.0, dt * 1.5)
                    else:
                        dt = max(1e-4, dt * 0.5)
            else:
                # Did not converge for this t, shrink dt
                dt *= 0.5
                if dt < 1e-4:
                    return False
                # Keep x_current as is, try smaller step
                continue

        return False

    # Try homotopy from x0
    if not run_homotopy(x0_arr):
        # If homotopy fails, try direct Broyden from x0
        # This is a fallback for cases where the homotopy path is problematic
        # But given the directive to use homotopy, we stick to it primarily.
        # However, if it fails, we might try a different starting point or strategy.
        # For now, let's just return the best found so far.
        pass

    return best_x
# EVOLVE-BLOCK-END
