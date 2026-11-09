import time
import math
import scipy.optimize

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    T0 = time.time()
    DEADLINE = 145.0
    TOL = 1e-8
    CALL_LIMIT = 480  # Target well under 500 to maximize score

    def elapsed_ok():
        return (time.time() - T0) < DEADLINE

    call_count = 0
    best_x = x0
    best_val = float('inf')

    def track(x, fx):
        nonlocal best_x, best_val
        if fx == fx:
            av = abs(fx)
            if av < best_val:
                best_val = av
                best_x = x

    def safe_f(x):
        nonlocal call_count
        if call_count >= CALL_LIMIT:
            return float('nan')
        call_count += 1
        try:
            return f(x)
        except Exception:
            return float('nan')

    # 1. Evaluate start
    f0 = safe_f(x0)
    track(x0, f0)
    if f0 == f0 and abs(f0) <= TOL:
        return x0

    # 2. Adaptive Bracketing Search with small steps first
    base_step = max(1e-3, abs(x0) * 1e-3, 1e-6)
    
    last_fwd_x, last_fwd_f = x0, f0
    last_bwd_x, last_bwd_f = x0, f0
    
    found_bracket = False
    bracket_a = bracket_b = bracket_fa = bracket_fb = None
    
    # Start with small steps (k from -20) to catch nearby roots
    # Then expand exponentially
    for k in range(-20, 40):
        if not elapsed_ok() or call_count >= CALL_LIMIT:
            break
        
        step = base_step * (2 ** k)
        if step < 1e-15:
            continue
        
        # Forward
        x_new = last_fwd_x + step
        f_new = safe_f(x_new)
        track(x_new, f_new)
        
        if f_new == f_new and last_fwd_f == last_fwd_f:
            if last_fwd_f * f_new < 0:
                found_bracket = True
                bracket_a, bracket_b = last_fwd_x, x_new
                bracket_fa, bracket_fb = last_fwd_f, f_new
                break
            if abs(f_new) <= TOL:
                return x_new
        
        last_fwd_x, last_fwd_f = x_new, f_new
        
        if not elapsed_ok() or call_count >= CALL_LIMIT:
            break
        
        # Backward
        x_new = last_bwd_x - step
        f_new = safe_f(x_new)
        track(x_new, f_new)
        
        if f_new == f_new and last_bwd_f == last_bwd_f:
            if last_bwd_f * f_new < 0:
                found_bracket = True
                bracket_a, bracket_b = x_new, last_bwd_x
                bracket_fa, bracket_fb = f_new, last_bwd_f
                break
            if abs(f_new) <= TOL:
                return x_new
        
        last_bwd_x, last_bwd_f = x_new, f_new

    # If we found a bracket, refine with safeguarded Newton/bisection
    if found_bracket:
        try:
            # Use scipy.optimize.brentq for robust refinement
            # Wrap f to track calls and respect limit
            def f_wrapper(x):
                return safe_f(x)
            
            root = scipy.optimize.brentq(f_wrapper, bracket_a, bracket_b, xtol=1e-14, rtol=1e-14, maxiter=100)
            f_root = safe_f(root)
            
            if f_root == f_root and abs(f_root) <= TOL:
                return root
            else:
                track(root, f_root)
                if best_val <= TOL:
                    return best_x
        except Exception:
            pass
    
    # Fallback: Return best_x found so far
    if best_x == best_x and best_val < float('inf'):
        return best_x
    return x0

# EVOLVE-BLOCK-END
