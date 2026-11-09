import math
import time
import scipy.optimize

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    max_calls = 480
    call_count = 0
    
    def time_ok():
        return (time.time() - start_time) < deadline and call_count < max_calls

    # Helper to call f and track count
    def f_call(x):
        nonlocal call_count
        call_count += 1
        return f(x)

    x = float(x0)
    fx = f_call(x)
    
    if not math.isfinite(fx):
        # If initial point is bad, we are in trouble. 
        # Try perturbing? No, task says x0 is a float starting point.
        # If f(x0) is inf, returning x0 is likely a failure unless f(x0) is actually 0? 
        # No, |inf| > 1e-8.
        # If we can't even evaluate, we can't solve. 
        # But usually x0 is fine.
        return x

    if abs(fx) <= 1e-8:
        return x

    # --- Phase 1: Find a Bracket ---
    # We try to find [a, b] such that f(a)*f(b) < 0.
    # Strategy: Check immediate neighbors, then expand.
    
    # Candidates for bracket endpoints
    # We start with a = x0, fa = fx.
    a = x
    fa = fx
    
    # Try right expansion
    h = 1.0
    b = None
    fb = None
    
    # Try a few small steps first to catch close roots
    # Then expand
    for i in range(40):
        if not time_ok():
            break
        
        # Check right
        b_candidate = a + h
        fb_candidate = f_call(b_candidate)
        
        if not math.isfinite(fb_candidate):
            h *= 2
            continue
            
        if fa * fb_candidate < 0:
            b = b_candidate
            fb = fb_candidate
            break
            
        # Check left (only if we haven't moved 'a' yet or if we are expanding from original x0)
        # Actually, simpler: Just expand a window [a-h, a+h] or just expand a and b.
        # Let's stick to a robust expansion:
        # If no sign change, we move the window.
        # If fa * fb_candidate > 0, the root is not in [a, b_candidate].
        # It could be to the left of a, or to the right of b_candidate.
        
        # Let's try a different strategy for bracketing:
        # Expand symmetrically from x0? Or just expand in one direction?
        # Polynomials can have roots anywhere.
        
        # Let's try: 
        # 1. f(x0)
        # 2. f(x0 + 1)
        # 3. f(x0 - 1)
        # 4. f(x0 + 2)
        # 5. f(x0 - 2)
        # ...
        
        # This requires tracking the best candidate bracket.
        
        # Let's restart the bracketing logic to be more systematic.
        pass

    # Re-implementing Bracketing more cleanly
    a = x0
    fa = fx
    # Try to find a sign change with x0 + d and x0 - d
    d = 1.0
    found = False
    
    for _ in range(30):
        if not time_ok():
            break
            
        # Try right
        xr = a + d
        fr = f_call(xr)
        if math.isfinite(fr):
            if fa * fr < 0:
                b = xr
                fb = fr
                found = True
                break
        
        # Try left
        xl = a - d
        fl = f_call(xl)
        if math.isfinite(fl):
            if fa * fl < 0:
                b = xl
                fb = fl
                found = True
                break
        
        # If no sign change with current 'a', move 'a' towards the side we just checked?
        # No, that's messy.
        # Let's just expand d.
        d *= 2.0
        
    if not found:
        # No bracket found.
        # Try a larger jump one last time
        d = 1e6
        xr = a + d
        fr = f_call(xr)
        if math.isfinite(fr) and fa * fr < 0:
            b = xr
            fb = fr
            found = True
        else:
            xl = a - d
            fl = f_call(xl)
            if math.isfinite(fl) and fa * fl < 0:
                b = xl
                fb = fl
                found = True

    if not found:
        # Extreme fallback: Secant without bracket
        x1 = x0 + 1e-6
        fx1 = f_call(x1)
        for _ in range(20):
            if not time_ok(): break
            if abs(fx) <= 1e-8: return x0
            if abs(fx1) <= 1e-8: return x1
            if fx1 == fx: break
            x_new = x0 - fx * (x0 - x1) / (fx - fx1)
            if not math.isfinite(x_new): break
            f_new = f_call(x_new)
            if not math.isfinite(f_new): break
            x0, fx = x1, fx1
            x1, fx1 = x_new, f_new
        return x0 if abs(fx) < abs(fx1) else x1

    # --- Phase 2: Refine using SciPy Brent's Method ---
    # Ensure a < b
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
        
    try:
        # xtol and rtol are tight to ensure |f(x)| is small
        # maxiter limits the number of function calls indirectly
        # brentq typically uses < 50 iterations for double precision
        root = scipy.optimize.brentq(f_call, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
        if math.isfinite(root):
            # Verify the result
            f_root = f_call(root)
            if math.isfinite(f_root) and abs(f_root) <= 1e-8:
                return root
            # If brentq returned a point but |f| > 1e-8, return it anyway 
            # as it's the best estimate, or fallback.
            # For polynomials, this is unlikely.
            return root
        else:
            return a # Fallback
    except Exception:
        # Fallback to manual Illinois/Secant-Bisection
        x_prev = a
        fx_prev = fa
        x_curr = b
        fx_curr = fb
        
        for _ in range(100):
            if not time_ok():
                break
            if abs(fx_curr) <= 1e-8:
                return x_curr
            if abs(fx_prev) <= 1e-8:
                return x_prev
                
            if abs(x_curr - x_prev) < 1e-12 * max(1.0, abs(x_curr), abs(x_prev)):
                return 0.5 * (x_curr + x_prev)
                
            if fx_curr != fx_prev:
                x_secant = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
            else:
                x_secant = 0.5 * (x_curr + x_prev)
                
            if min(x_prev, x_curr) < x_secant < max(x_prev, x_curr):
                x_next = x_secant
            else:
                x_next = 0.5 * (x_curr + x_prev)
                
            f_next = f_call(x_next)
            
            if not math.isfinite(f_next):
                 x_next = 0.5 * (x_curr + x_prev)
                 f_next = f_call(x_next)
                 
            if abs(f_next) <= 1e-8:
                return x_next
                 
            if fx_prev * f_next < 0:
                x_curr = x_next
                fx_curr = f_next
            else:
                x_prev = x_next
                fx_prev = f_next
                
        return x_curr if abs(fx_curr) < abs(fx_prev) else x_prev
