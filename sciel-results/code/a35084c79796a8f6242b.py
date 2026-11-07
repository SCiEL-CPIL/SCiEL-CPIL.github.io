"""Root finder for 3 fixed 2D systems using Homotopy/Continuation Method."""
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

    def F_t(x, t):
        """
        Homotopy function: F_t(x) = t * F(x) + (1 - t) * (x - x0)
        We want to find x such that F_t(x) = 0.
        At t=0, F_0(x) = x - x0, so root is x0.
        At t=1, F_1(x) = F(x), so root is the target.
        """
        Fx, _ = safe_eval(x)
        if Fx is None:
            return None, float('inf')
        Ft = t * Fx + (1 - t) * (x - x0_arr)
        norm_Ft = np.linalg.norm(Ft)
        return Ft, norm_Ft

    # Broyden's method for tracking the homotopy path
    # We maintain an approximation J of the Jacobian of F_t with respect to x.
    # Note: J_t = t * J_F + (1-t) * I.
    # We can update J_F using Broyden's formula applied to the residual F_t.
    # Let's just maintain J_t directly or J_F? 
    # It's easier to maintain J_F and compute J_t on the fly, 
    # but Broyden's update requires the Jacobian of the function being solved.
    # Let's maintain J_t directly.
    
    # Initial state at t=0
    # F_0(x) = x - x0. J_0 = I.
    x = x0_arr.copy()
    J = np.eye(2)
    t = 0.0
    
    best_x = x0_arr.copy()
    best_norm = float('inf')
    
    # Check initial point for the original problem (just in case x0 is already a root)
    Fx_init, norm_init = safe_eval(x0_arr)
    if Fx_init is not None and norm_init <= 1e-8:
        return x0_arr
    if Fx_init is not None and norm_init < best_norm:
        best_norm = norm_init
        best_x = x0_arr.copy()

    # Homotopy steps
    # We want to go from t=0 to t=1.
    # Adaptive step size: start with small dt, increase if successful.
    dt = 0.1
    max_dt = 0.5
    min_dt = 0.01
    
    # We will iterate until t >= 1 or we run out of time/calls
    while t < 1.0:
        if time.time() - start_time > time_limit:
            break
        if call_counter[0] >= max_calls:
            break
            
        # Ensure we don't overshoot 1.0
        dt = min(dt, 1.0 - t)
        
        # Try to solve F_{t+dt}(x) = 0 using Broyden's method, starting from current x
        # Current J is approx of J_t.
        # We need to solve for x_new such that F_{t+dt}(x_new) = 0.
        
        t_new = t + dt
        x_curr = x.copy()
        J_curr = J.copy() # J is approx of J_t
        
        # We need to initialize the Broyden loop for the new t.
        # The function is G(x) = F_{t_new}(x).
        # Its Jacobian is J_G = t_new * J_F + (1-t_new) * I.
        # We don't have J_F directly, we have J_t = t * J_F + (1-t) * I.
        # So J_F = (J_t - (1-t)*I) / t (if t > 0).
        # This is risky if t is small.
        
        # Alternative: Update J_t to J_{t_new} analytically?
        # J_{t_new} = t_new * J_F + (1-t_new) * I
        # J_t = t * J_F + (1-t) * I
        # If we assume J_F is constant over the small step (which is an approximation),
        # we can estimate J_{t_new}.
        # However, Broyden's method updates the Jacobian approximation based on the function values.
        # It is more robust to just treat G(x) = F_{t_new}(x) as a new function and use the previous x as start.
        # We can initialize J_G using the relation above if t > 0, else I.
        
        if t > 1e-6:
            # Estimate J_F from current J (approx of J_t)
            J_F_est = (J_curr - (1 - t) * np.eye(2)) / t
            J_G_init = t_new * J_F_est + (1 - t_new) * np.eye(2)
        else:
            # At t=0, J_0 = I. J_F is unknown. 
            # But F_0(x) = x - x0. 
            # For t_new small, J_G is close to I.
            J_G_init = np.eye(2)
            
        # Broyden's loop for fixed t_new
        converged = False
        for _ in range(20): # Max iterations per homotopy step
            if time.time() - start_time > time_limit:
                break
            if call_counter[0] >= max_calls:
                break
                
            # Evaluate G(x_curr) = F_{t_new}(x_curr)
            Fx_curr, _ = safe_eval(x_curr)
            if Fx_curr is None:
                break
            G_curr = t_new * Fx_curr + (1 - t_new) * (x_curr - x0_arr)
            norm_G = np.linalg.norm(G_curr)
            
            # Check convergence of the homotopy step
            if norm_G < 1e-10:
                converged = True
                break
                
            # Broyden step
            try:
                s = np.linalg.solve(J_G_init, -G_curr)
            except np.linalg.LinAlgError:
                try:
                    s = np.linalg.lstsq(J_G_init, -G_curr, rcond=None)[0]
                except:
                    s = -G_curr
                    
            if not np.all(np.isfinite(s)):
                break
                
            # Line search for G(x)
            alpha = 1.0
            x_new = None
            G_new = None
            norm_new = float('inf')
            
            for _ in range(15):
                x_trial = x_curr + alpha * s
                Fx_trial, _ = safe_eval(x_trial)
                if Fx_trial is None:
                    alpha *= 0.5
                    continue
                    
                G_trial = t_new * Fx_trial + (1 - t_new) * (x_trial - x0_arr)
                norm_trial = np.linalg.norm(G_trial)
                
                if norm_trial < (1 - 1e-4 * alpha) * norm_G:
                    x_new = x_trial
                    G_new = G_trial
                    norm_new = norm_trial
                    break
                else:
                    alpha *= 0.5
                    
            if x_new is None:
                break
                
            # Update Jacobian approximation J_G_init
            delta_x = x_new - x_curr
            delta_G = G_new - G_curr
            denom = np.dot(delta_x, delta_x)
            
            if denom > 1e-12:
                J_G_init = J_G_init + np.outer(delta_G - J_G_init @ delta_x, delta_x) / denom
                
            x_curr = x_new
            
            # Check if we converged to the original root (t_new close to 1)
            if t_new > 0.99 and norm_new <= 1e-8:
                # We found a root for F(x)
                # But we need to be careful: norm_new is ||F_{t_new}(x)||, not ||F(x)||
                # Since t_new is close to 1, ||F_{t_new}(x)|| approx ||F(x)|| + (1-t_new)||x-x0||
                # If (1-t_new) is small, this is a good check.
                # Let's do a final check on F(x) directly if we are close to t=1.
                Fx_check, norm_F_check = safe_eval(x_new)
                if Fx_check is not None and norm_F_check <= 1e-8:
                    best_x = x_new.copy()
                    best_norm = norm_F_check
                    converged = True
                    break
                elif Fx_check is not None and norm_F_check < best_norm:
                    best_x = x_new.copy()
                    best_norm = norm_F_check

        if not converged:
            # Step failed, reduce dt and try again
            dt *= 0.5
            if dt < min_dt:
                # If we can't take small steps, break
                break
            continue
        
        # Step successful
        x = x_curr
        # Update J to be approx of J_{t_new}
        # We have J_G_init which is approx of J_{t_new}
        J = J_G_init
        t = t_new
        
        # Increase dt if stable
        if dt < max_dt:
            dt *= 1.2
            
        # If we reached t=1, check the final root
        if t >= 1.0 - 1e-6:
             Fx_final, norm_final = safe_eval(x)
             if Fx_final is not None and norm_final <= 1e-8:
                 best_x = x.copy()
                 best_norm = norm_final
                 break
             elif Fx_final is not None and norm_final < best_norm:
                 best_x = x.copy()
                 best_norm = norm_final

    # Final polish if we have a candidate
    if best_norm < 1e-4:
        # Use a few Newton steps with finite difference Jacobian to polish
        for _ in range(5):
            if call_counter[0] >= max_calls:
                break
            Fx, norm_F = safe_eval(best_x)
            if Fx is None:
                break
            if norm_F <= 1e-8:
                break
                
            # Finite difference Jacobian
            h = 1e-5
            J_fd = np.zeros((2, 2))
            for i in range(2):
                x_plus = best_x.copy()
                x_plus[i] += h
                F_plus, _ = safe_eval(x_plus)
                if F_plus is None:
                    J_fd = None
                    break
                J_fd[:, i] = (F_plus - Fx) / h
                
            if J_fd is None:
                break
                
            try:
                s = np.linalg.solve(J_fd, -Fx)
                best_x = best_x + s
            except:
                break

    return best_x
# EVOLVE-BLOCK-END
