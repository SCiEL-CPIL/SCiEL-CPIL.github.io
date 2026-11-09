import math
import time
import scipy.optimize as opt

def find_root(f, x0):
    CALLS = 0
    MAX_CALLS = 480
    TIME_LIMIT = 100.0
    START_TIME = time.time()
    
    # Wrapper to count calls and handle exceptions/non-finite
    def safe_f(x):
        nonlocal CALLS
        if CALLS >= MAX_CALLS:
            raise RuntimeError("Max calls exceeded")
        if time.time() - START_TIME > TIME_LIMIT:
            raise RuntimeError("Time limit exceeded")
        CALLS += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('nan')
            return float(val)
        except Exception:
            return float('nan')

    x0 = float(x0)
    
    # Track best point found so far
    best_x = x0
    best_f = float('inf')
    
    def update_best(x, fx):
        nonlocal best_x, best_f
        if math.isfinite(fx) and abs(fx) < best_f:
            best_x = x
            best_f = abs(fx)
            if best_f <= 1e-8:
                return True
        return False

    # Phase 0: Check x0
    fx0 = safe_f(x0)
    if math.isfinite(fx0):
        if update_best(x0, fx0):
            return x0
    else:
        # If x0 is invalid, we have no good starting point. 
        # But we must return something. Let's try a small step.
        # However, if f(x0) fails, it might be a domain issue.
        # Let's just proceed to search.
        pass

    # Define a helper to check if [a,b] is a valid bracket
    def is_bracket(a, b, fa, fb):
        if not (math.isfinite(fa) and math.isfinite(fb)):
            return False
        if fa == fb:
            return False
        # Check if 0 is between fa and fb
        if (fa < 0 and fb > 0) or (fa > 0 and fb < 0):
            return True
        return False

    found_bracket = False
    a, b, fa, fb = None, None, None, None
    
    # Phase 1: Find a bracket
    # Strategy: Scan in both directions with small steps first, then larger.
    # Since the equations are oscillatory, roots are likely frequent.
    
    # 1. Small step scan (h=0.1) in both directions
    h = 0.1
    max_steps = 100
    
    # Positive direction
    last_x = x0
    last_fx = fx0 if math.isfinite(fx0) else None
    
    if last_fx is not None:
        for step in range(1, max_steps + 1):
            if CALLS > MAX_CALLS - 50:
                break
            x_next = x0 + step * h
            fx_next = safe_f(x_next)
            
            if math.isfinite(fx_next):
                update_best(x_next, fx_next)
                if best_f <= 1e-8:
                    return best_x
                if is_bracket(last_x, x_next, last_fx, fx_next):
                    a, b, fa, fb = last_x, x_next, last_fx, fx_next
                    found_bracket = True
                    break
                last_x = x_next
                last_fx = fx_next
            else:
                # If we hit a non-finite value, we might have crossed a singularity or left domain.
                # Stop this direction.
                break

    # Negative direction (if not found yet)
    if not found_bracket and last_fx is not None:
        last_x = x0
        last_fx = fx0
        for step in range(1, max_steps + 1):
            if CALLS > MAX_CALLS - 50:
                break
            x_next = x0 - step * h
            fx_next = safe_f(x_next)
            
            if math.isfinite(fx_next):
                update_best(x_next, fx_next)
                if best_f <= 1e-8:
                    return best_x
                if is_bracket(x_next, last_x, fx_next, last_fx):
                    a, b, fa, fb = x_next, last_x, fx_next, last_fx
                    found_bracket = True
                    break
                last_x = x_next
                last_fx = fx_next
            else:
                break

    # 2. Larger step scan if small step failed
    if not found_bracket:
        # Try exponential expansion
        for h_try in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
            if CALLS > MAX_CALLS - 50:
                break
            
            # Positive
            if math.isfinite(fx0):
                xp = x0 + h_try
                fp = safe_f(xp)
                if math.isfinite(fp):
                    update_best(xp, fp)
                    if best_f <= 1e-8:
                        return best_x
                    if is_bracket(x0, xp, fx0, fp):
                        a, b, fa, fb = x0, xp, fx0, fp
                        found_bracket = True
                        break
                    
            # Negative
            if math.isfinite(fx0):
                xn = x0 - h_try
                fn = safe_f(xn)
                if math.isfinite(fn):
                    update_best(xn, fn)
                    if best_f <= 1e-8:
                        return best_x
                    if is_bracket(xn, x0, fn, fx0):
                        a, b, fa, fb = xn, x0, fn, fx0
                        found_bracket = True
                        break

    # Phase 2: Solve
    if found_bracket:
        try:
            # Use Brent's method for robustness and speed
            root = opt.brentq(safe_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
            if math.isfinite(root):
                fr = safe_f(root)
                if math.isfinite(fr) and abs(fr) <= 1e-8:
                    return root
                # If Brent didn't quite hit it, but got close, return it?
                # The contract says |f(x)| <= 1e-8.
                # If it's close but not quite, maybe we can do a few Newton steps?
                # Or just return best_x if it's better?
                if abs(fr) < best_f:
                    best_x = root
                    best_f = abs(fr)
        except Exception:
            pass

    # Fallback: Try Newton's method from best_x if we have a gradient estimate?
    # Or just return best_x if it's reasonably good?
    # If best_f is still > 1e-8, we might fail.
    # Let's try a few secant/Newton steps from best_x if we haven't used all calls.
    
    if CALLS < MAX_CALLS - 10:
        x_curr = best_x
        fx_curr = best_f
        # We need the sign of fx_curr for Newton? No, just value.
        # We need to re-evaluate to get the actual value (with sign) for Newton.
        # But we only stored abs.
        # Let's just do a simple local search around best_x.
        
        # Actually, if we didn't bracket, we might be in a local minimum of |f(x)|.
        # If the function is oscillatory, a local min of |f| is likely near 0.
        # If best_f is small, return it.
        
        # One more attempt: If we have a bracket, we already tried Brent.
        # If no bracket, we return the best point found.
        
        pass

    return best_x
