"""Root finding for ill-conditioned equations: Global Homotopy Path-Following."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin below 120s limit
    MAX_EVALS = 450   # Safety margin below 500 limit
    
    # State tracking
    eval_count = 0
    best_x = float(x0)
    best_f = None
    try:
        best_f = f(best_x)
        eval_count += 1
    except Exception:
        best_f = float('inf')

    def check_budget():
        nonlocal eval_count
        if eval_count > MAX_EVALS or (time.time() - start_time) > DEADLINE:
            return True
        return False

    def safe_eval(x):
        nonlocal eval_count, best_x, best_f
        if check_budget():
            return None
        try:
            val = f(x)
            if not math.isfinite(val):
                return None
            eval_count += 1
            if abs(val) < abs(best_f):
                best_f = val
                best_x = x
            return val
        except Exception:
            return None

    # 1. Initial Check
    if best_f is not None and abs(best_f) <= 1e-8:
        return best_x

    # 2. Homotopy Strategy
    # We solve (1-t)*f(x) + t*x = 0
    # At t=0, root is 0 (assuming f(0) is defined, though we start from x0 logic)
    # Actually, standard homotopy H(x,t) = (1-t)f(x) + t*(x - x_target) is common.
    # But here we want a root of f. 
    # Let's use H(x,t) = (1-t)f(x) + t*x.
    # At t=1, H(x,1) = x = 0 => root is 0.
    # At t=0, H(x,0) = f(x) = 0 => root is target.
    # So we start at t=1 (root=0) and move to t=0.
    
    # However, if f(0) is huge or undefined, this might be hard.
    # Let's try to start from the provided x0 if it's close, or use 0 as anchor.
    # A more robust anchor: Use the provided x0 as the "target" for the homotopy?
    # No, the standard "continuation" method solves f_t(x) = (1-t)f(x) + t*(x-x0) = 0.
    # At t=1, x=x0 is a root.
    # At t=0, f(x)=0.
    # This guarantees we start at a known point (x0) and deform to the problem.
    
    x_current = float(x0)
    t = 1.0
    
    # If x0 is already a root, we are done.
    # If not, we step t down to 0.
    
    # Adaptive step size for t
    dt = 0.25
    min_dt = 1e-4
    
    while t > 1e-6:
        if check_budget():
            break
            
        # Try to solve (1-t)f(x) + t*(x - x0) = 0 for x
        # Let g(x) = (1-t)f(x) + t*(x - x0)
        # We use Newton's method on g(x), starting from x_current
        
        # Derivative of g(x) w.r.t x:
        # g'(x) = (1-t)f'(x) + t
        # We approximate f'(x) via finite difference.
        
        solved = False
        x_try = x_current
        
        for _ in range(20): # Max Newton steps for this t
            if check_budget():
                break
                
            # Evaluate g(x_try)
            f_val = safe_eval(x_try)
            if f_val is None:
                break
            
            g_val = (1 - t) * f_val + t * (x_try - x0)
            
            if abs(g_val) < 1e-12:
                # Converged for this t
                solved = True
                break
                
            # Estimate derivative
            h = 1e-6 * max(1.0, abs(x_try))
            f_val_plus = safe_eval(x_try + h)
            if f_val_plus is None:
                break
            f_prime = (f_val_plus - f_val) / h
            g_prime = (1 - t) * f_prime + t
            
            if g_prime == 0:
                break
                
            step = -g_val / g_prime
            
            # Damping to ensure we don't jump too far
            # Limit step to 10% of distance to x0 or absolute 10
            max_step = max(10.0, 0.1 * abs(x_try - x0))
            if abs(step) > max_step:
                step = math.copysign(max_step, step)
                
            x_new = x_try + step
            
            # Check if x_new is valid (finite)
            if not math.isfinite(x_new):
                break
                
            # Optional: Check if |g| decreases
            # For simplicity, we just take the step if it's not exploding
            if abs(x_new - x_try) < 1e-15:
                solved = True
                break
                
            x_try = x_new
        
        if solved:
            x_current = x_try
            # Decrease t
            t -= dt
            if t < 0:
                t = 0
            # Increase step size slightly if we are stable?
            # For robustness, keep dt constant or reduce if it failed.
            # Let's reduce dt if we failed to solve
        else:
            # Failed to converge for this t
            # Reduce step size
            dt = max(dt * 0.5, min_dt)
            if dt <= min_dt:
                # If we can't even take a tiny step, we might be stuck.
                # Try a different strategy: Secant method on the original f from x_current
                # Or just break and return best.
                break

    # 3. Final Refinement with Secant/Newton on original f
    # The homotopy gets us close. Now polish.
    x_final = x_current
    
    # Check if we are already good
    f_final = safe_eval(x_final)
    if f_final is not None and abs(f_final) <= 1e-8:
        return x_final

    # Polishing phase: Newton with safeguarding
    x_polish = x_final
    for _ in range(30):
        if check_budget():
            break
        fx = safe_eval(x_polish)
        if fx is None:
            break
        if abs(fx) <= 1e-8:
            return x_polish
        
        h = 1e-6 * max(1.0, abs(x_polish))
        fx_plus = safe_eval(x_polish + h)
        if fx_plus is None:
            break
        dfx = (fx_plus - fx) / h
        if dfx == 0:
            break
        
        step = -fx / dfx
        # Clip step
        if abs(step) > 1.0:
            step = math.copysign(1.0, step)
            
        x_new = x_polish + step
        if not math.isfinite(x_new):
            break
            
        x_polish = x_new

    return best_x
# EVOLVE-BLOCK-END
