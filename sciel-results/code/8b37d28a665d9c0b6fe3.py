"""Root finder for 3 fixed 2D systems using Broyden's Method with Adaptive Line Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _safe_eval(F, x):
    """Safely evaluate F at x, returning (Fx, norm_Fx, is_valid)."""
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,):
            Fx = Fx.reshape(2)
        if not np.all(np.isfinite(Fx)):
            return None, float('inf'), False
        norm_Fx = np.linalg.norm(Fx)
        return Fx, norm_Fx, True
    except Exception:
        return None, float('inf'), False

def _broyden_step(F, x, J, max_iter=50, tol=1e-10, call_counter=None):
    """
    Broyden's method (Good variant) with adaptive line search.
    x: current point
    J: current Jacobian estimate (2x2)
    call_counter: list with one int to track calls
    """
    Fx, norm_F, valid = _safe_eval(F, x)
    if call_counter is not None:
        call_counter[0] += 1
    if not valid:
        return x, float('inf'), J, call_counter
    
    if norm_F <= tol:
        return x, norm_F, J, call_counter

    for _ in range(max_iter):
        if call_counter is not None and call_counter[0] > 480: # Safety margin
            break
            
        # Solve J * s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse or reset J
            try:
                s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                s = -Fx # Gradient descent step as fallback
                
        if not np.all(np.isfinite(s)):
            # If step is bad, try a small gradient step
            s = -0.1 * Fx
            if not np.all(np.isfinite(s)):
                break

        # Adaptive Line Search
        # We want to find t in (0, 1] such that ||F(x + t*s)|| < (1 - c*t) * ||F(x)||
        # Or simply minimize ||F|| in the direction s.
        # Standard backtracking:
        alpha = 1.0
        c = 1e-4 # Armijo condition constant
        max_backtracks = 10
        
        x_new = x
        F_new = None
        norm_new = float('inf')
        
        for _ in range(max_backtracks):
            x_trial = x + alpha * s
            F_trial, norm_trial, valid_trial = _safe_eval(F, x_trial)
            if call_counter is not None:
                call_counter[0] += 1
                
            if not valid_trial:
                alpha *= 0.5
                continue
                
            # Check if we made progress (reduction in norm)
            # Simple sufficient decrease
            if norm_trial < norm_F:
                x_new = x_trial
                F_new = F_trial
                norm_new = norm_trial
                break
            else:
                alpha *= 0.5
                
        if F_new is None:
            # No improvement found with line search, maybe shrink step size drastically
            # or break if alpha is too small
            if alpha < 1e-6:
                break
            # Force a small step if needed to avoid stagnation? 
            # For now, break if no progress
            break

        # Update Jacobian (Broyden Good)
        # J_new = J + (F_new - Fx - J*s) * (s^T) / (s^T * s)
        try:
            delta_F = F_new - Fx
            delta_x = x_new - x
            # s^T s is alpha^2 * s^T s
            denom = np.dot(delta_x, delta_x)
            if denom > 1e-12:
                J = J + np.outer(delta_F - J @ delta_x, delta_x) / denom
            else:
                # If step was tiny, just keep J
                pass
        except Exception:
            # If update fails, keep old J
            pass

        x = x_new
        Fx = F_new
        norm_F = norm_new
        
        if norm_F <= tol:
            break

    return x, norm_F, J, call_counter

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
    
    # Strategy: 
    # 1. Try Broyden from x0 with identity Jacobian.
    # 2. If it fails or is slow, try other starting points.
    # Broyden is very cheap per iteration (1-2 calls). 
    # We can afford many starts if they fail fast, or fewer if they converge.
    
    # Generate starts
    starts = [x0_arr, np.zeros(2)]
    
    # Add points on a grid/circle to cover basins of attraction
    radii = [0.5, 1.0, 2.0, 5.0, 10.0]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    for r in radii:
        for a in angles:
            starts.append(np.array([r * np.cos(a), r * np.sin(a)]))
            
    # Random points
    rng = np.random.RandomState(42)
    for _ in range(20):
        starts.append(rng.uniform(-20, 20, 2))
        
    # Shuffle starts slightly to avoid always trying x0 first if it's bad? 
    # No, x0 is usually a good hint. But if x0 is far, it might waste calls.
    # Let's stick to the order: x0, origin, then others.
    
    J_init = np.eye(2)
    
    for x_start in starts:
        if time.time() - start_time > time_limit:
            break
            
        if call_counter[0] >= max_calls - 50: # Reserve 50 calls for safety
            break
            
        # Initialize Jacobian. 
        # We could estimate it with 2 calls (forward diff) to improve first step,
        # but Identity is often good enough for Broyden to self-correct.
        # Let's try Identity first.
        
        x_res, norm_res, _, _ = _broyden_step(F, x_start, J_init, max_iter=100, tol=1e-10, call_counter=call_counter)
        
        if norm_res < best_norm:
            best_norm = norm_res
            best_x = x_res
            
        if best_norm <= 1e-8:
            break
            
        # Optional: If Broyden stalled, we could try a different J initialization 
        # but Broyden updates J, so reusing J from a failed start is risky.
        # We'll just move to the next start with Identity J.
        
    return best_x
# EVOLVE-BLOCK-END
