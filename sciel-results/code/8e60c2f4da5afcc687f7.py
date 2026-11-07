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

    best_x = x0_arr.copy()
    best_norm = float('inf')
    
    # Homotopy path: F_t(x) = t * F(x) + (1-t) * (x - x0)
    # We want to find x(t) such that F_t(x(t)) = 0.
    # At t=0, x(0) = x0.
    # At t=1, x(1) is a root of F(x) = 0.
    
    # We will step t from 0 to 1 in increments.
    # For each t, we solve for x using Newton's method on F_t.
    # The Jacobian of F_t is: J_t = t * J_F + (1-t) * I
    # We don't have J_F, so we use Broyden's method to approximate J_t for each t.
    # Since t changes, the "Jacobian" changes, but we can warm-start the Broyden update
    # from the previous t's approximation, adjusted for the change in t.
    
    # Let H_t approximate J_{F_t}.
    # J_{F_{t+dt}} = (t+dt) J_F + (1-t-dt) I
    # J_{F_t} = t J_F + (1-t) I
    # Difference = dt * (J_F - I)
    # We can update H by adding dt * (H - I) approximately?
    # Actually, simpler: just run Broyden on F_t with a fresh or updated H.
    # Given the small dimension, a fresh H=I for each t might be robust enough,
    # but reusing H is faster. Let's try reusing H and letting Broyden correct it.
    
    x_current = x0_arr.copy()
    
    # Initial evaluation to check if x0 is already a root
    Fx, norm_F = safe_eval(x_current)
    if Fx is not None:
        best_norm = norm_F
        if norm_F <= 1e-8:
            return best_x
            
    # We will use a sequence of t values.
    # Linear steps might be too coarse if the path is curved.
    # Adaptive steps based on convergence of the inner solver.
    
    t = 0.0
    dt = 0.1  # Initial step size
    
    # Hessian approximation for the current F_t
    # At t=0, F_0(x) = x - x0, so J_0 = I.
    H = np.eye(2)
    
    max_inner_iters = 20
    max_outer_steps = 50
    
    for outer_step in range(max_outer_steps):
        if time.time() - start_time > time_limit:
            break
        if call_counter[0] >= max_calls:
            break
            
        # Increase t
        t_new = min(1.0, t + dt)
        
        # Define F_t
        def F_t(x, t_val=t_new):
            Fx_orig, norm_orig = safe_eval(x)
            if Fx_orig is None:
                return None, float('inf')
            # F_t(x) = t * F(x) + (1-t) * (x - x0)
            Fx_t = t_val * Fx_orig + (1.0 - t_val) * (x - x0_arr)
            norm_t = np.linalg.norm(Fx_t)
            return Fx_t, norm_t
        
        # Solve F_t(x) = 0 using Broyden's method
        # Start from x_current
        x = x_current.copy()
        Fx_t, norm_t = F_t(x)
        
        if Fx_t is None:
            # If evaluation fails, shrink dt and retry
            dt *= 0.5
            if dt < 1e-4:
                break
            continue
            
        # Broyden iterations
        converged = False
        for inner_iter in range(max_inner_iters):
            if time.time() - start_time > time_limit:
                break
            if call_counter[0] >= max_calls:
                break
                
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
            
            # Track best
            if norm_t < best_norm:
                best_norm = norm_t
                best_x = x.copy()
                
            if norm_t <= 1e-8:
                converged = True
                break
                
        if converged:
            # Check if we've reached t=1
            if t_new >= 1.0:
                # We are at a root of F(x)
                # Do a final polish with pure F(x) to ensure precision
                # The norm_t is ||F_1(x)|| = ||F(x)||, so it's already good.
                # But let's do a few more Broyden steps on F(x) directly to be safe
                # and to update H to J_F.
                
                # Switch to pure F
                x_polish = x.copy()
                Fx_polish, norm_polish = safe_eval(x_polish)
                if Fx_polish is not None:
                    best_norm = norm_polish
                    best_x = x_polish.copy()
                    
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
                            
                return best_x
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
                break
            # Keep x_current as is, try smaller step
            continue

    return best_x
# EVOLVE-BLOCK-END
