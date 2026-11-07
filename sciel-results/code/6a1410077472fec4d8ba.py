"""Root finder for 3 fixed 2D systems using Multi-Start Newton with Hammersley Sequence and Central Differences."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _safe_eval(F, x):
    """
    Safely evaluate F at x.
    Returns (Fx, norm_Fx, is_valid).
    """
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,):
            # Attempt to reshape if it's a flat array of size 2
            if Fx.size == 2:
                Fx = Fx.reshape(2)
            else:
                return None, float('inf'), False
        if not np.all(np.isfinite(Fx)):
            return None, float('inf'), False
        norm_Fx = float(np.linalg.norm(Fx))
        return Fx, norm_Fx, True
    except Exception:
        return None, float('inf'), False

def _estimate_jacobian_central(F, x, h=1e-5, call_counter=None):
    """
    Estimate the Jacobian using central differences.
    Requires 4 function evaluations.
    """
    J = np.zeros((2, 2))
    for j in range(2):
        xp = x.copy()
        xm = x.copy()
        xp[j] += h
        xm[j] -= h
        
        Fxp, _, valid_p = _safe_eval(F, xp)
        if call_counter is not None:
            call_counter[0] += 1
        Fxm, _, valid_m = _safe_eval(F, xm)
        if call_counter is not None:
            call_counter[0] += 1
            
        if valid_p and valid_m:
            J[:, j] = (Fxp - Fxm) / (2 * h)
        else:
            # Fallback to forward difference if central fails
            Fx0, _, valid_0 = _safe_eval(F, x)
            if call_counter is not None:
                call_counter[0] += 1
            if valid_p and valid_0:
                J[:, j] = (Fxp - Fx0) / h
            elif valid_m and valid_0:
                J[:, j] = (Fx0 - Fxm) / h
            else:
                # If all fail, return singular matrix to trigger error handling
                J[:, j] = np.array([1.0, 0.0]) if j == 0 else np.array([0.0, 1.0])
    return J

def _newton_step(F, x, call_counter, max_iter=20, tol=1e-12, time_limit=None, start_time=None):
    """
    Perform Newton's method with central difference Jacobian estimation.
    Includes damping if the step increases the norm.
    """
    # Initial evaluation
    Fx, norm_F, valid = _safe_eval(F, x)
    if call_counter is not None:
        call_counter[0] += 1
        
    if not valid:
        return x, float('inf')
    
    if norm_F <= tol:
        return x, norm_F

    for _ in range(max_iter):
        # Check time limit
        if time_limit is not None and time.time() > time_limit:
            break
            
        # Check call limit
        if call_counter is not None and call_counter[0] > 480:
            break

        # Estimate Jacobian
        J = _estimate_jacobian_central(F, x, h=1e-5, call_counter=call_counter)
        
        # Check if Jacobian is singular
        try:
            det_J = np.linalg.det(J)
            if np.abs(det_J) < 1e-12:
                # Add small regularization
                J += 1e-8 * np.eye(2)
        except np.linalg.LinAlgError:
            pass

        # Solve J * s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse
            try:
                s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                s = -Fx # Fallback to gradient descent

        if not np.all(np.isfinite(s)):
            break

        # Damped Newton Step
        # We want to reduce ||F(x)||. 
        # Try full step first.
        alpha = 1.0
        x_new = x
        F_new = None
        norm_new = float('inf')
        best_alpha = 0.0
        best_norm = norm_F

        # Backtracking line search
        for _ in range(10):
            x_trial = x + alpha * s
            F_trial, norm_trial, valid_trial = _safe_eval(F, x_trial)
            if call_counter is not None:
                call_counter[0] += 1
                
            if not valid_trial:
                alpha *= 0.5
                continue

            if norm_trial < best_norm:
                best_norm = norm_trial
                best_alpha = alpha
                x_new = x_trial
                F_new = F_trial
                norm_new = norm_trial
                
                if norm_trial <= tol:
                    break
            else:
                alpha *= 0.5
                if alpha < 1e-6:
                    break

        if F_new is None:
            # No improvement found
            break

        x = x_new
        Fx = F_new
        norm_F = norm_new
        
        if norm_F <= tol:
            break

    return x, norm_F

def _generate_hammersley_starts(n_starts, max_radius=20.0):
    """
    Generate starting points using a Hammersley-like sequence (low discrepancy)
    in polar coordinates (log radius, angle).
    """
    starts = []
    # Include origin and x0-like points implicitly by caller, but here we generate diverse points
    # Use a simple Halton sequence for 2D (log_r, theta)
    
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    # We want to cover a range of radii from 1e-2 to max_radius
    # log10(1e-2) = -2, log10(20) ~ 1.3
    # Let's map [0, 1) to [-2, 1.5]
    
    for i in range(1, n_starts + 1):
        # Halton base 2 for radius, base 3 for angle
        u_r = halton(i, 2)
        u_a = halton(i, 3)
        
        # Map u_r to log radius
        log_r_min = -2.0
        log_r_max = 1.5
        log_r = log_r_min + u_r * (log_r_max - log_r_min)
        r = 10 ** log_r
        
        # Map u_a to angle
        theta = u_a * 2 * np.pi
        
        x = np.array([r * np.cos(theta), r * np.sin(theta)])
        starts.append(x)
        
    return starts

def find_root_system(F, x0):
    start_time = time.time()
    time_limit = 145.0  # Safety margin from 150s
    
    x0_arr = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0_arr)):
        x0_arr = np.zeros(2)
        
    # Global call counter for this system
    call_counter = [0]
    max_calls = 500
    
    best_x = x0_arr
    best_norm = float('inf')
    
    # 1. Evaluate x0 first
    Fx0, norm0, valid0 = _safe_eval(F, x0_arr)
    call_counter[0] += 1
    if valid0:
        best_norm = norm0
        best_x = x0_arr
        if best_norm <= 1e-8:
            return best_x

    # 2. Generate diverse starting points
    # We reserve some calls for the Newton iterations themselves.
    # Each Newton iteration costs ~5 calls (1 for F, 4 for J).
    # If we allow 10 iterations per start, that's ~50 calls.
    # With 500 calls, we can afford ~10 starts if they fail, or fewer if they converge.
    
    n_starts = 15
    starts = _generate_hammersley_starts(n_starts, max_radius=50.0)
    
    # Add origin and x0 again in case they were skipped or failed
    starts.append(np.zeros(2))
    starts.append(x0_arr)
    
    # Shuffle with a fixed seed for reproducibility but diversity in order
    rng = np.random.RandomState(42)
    # We don't shuffle fully because x0 is usually good, but we interleave
    # Let's just iterate through them
    
    for x_start in starts:
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # Check calls
        if call_counter[0] >= max_calls - 20: # Reserve 20 calls for safety
            break
        
        # Run Newton from this start
        # Limit iterations to save calls if it's not converging fast
        x_res, norm_res = _newton_step(F, x_start, call_counter, max_iter=15, tol=1e-10, time_limit=start_time + time_limit, start_time=start_time)
        
        if norm_res < best_norm:
            best_norm = norm_res
            best_x = x_res
            
        if best_norm <= 1e-8:
            break
            
    return best_x
# EVOLVE-BLOCK-END
