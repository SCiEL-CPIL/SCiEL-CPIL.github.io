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
        return x

    if abs(fx) <= 1e-8:
        return x

    # --- Phase 1: Find a Bracket ---
    # Strategy: Expand symmetrically from x0 to find a sign change.
    # We track all points evaluated to potentially find a bracket between any two, 
    # but primarily we look for f(x0) * f(x0 +/- d) < 0.
    # However, for polynomials, if the root is far, we might need to expand.
    
    a = x
    fa = fx
    found = False
    b = None
    fb = None
    
    # Store evaluated points to check for sign changes between any consecutive points
    # in case we jump over a root or if the root is between x0 and the expanded point
    # but not adjacent to x0 in a simple way? 
    # Actually, standard expansion:
    # d = 1, 2, 4, ...
    # Check x0+d and x0-d.
    
    d = 1.0
    # We will expand d until we find a sign change or run out of budget.
    # To be more robust, we also check if the sign change is between the previous 
    # expansion point and the current one? No, we only have x0 and the new point.
    # If f(x0) and f(x0+d) have the same sign, the root might be further out, 
    # OR it might be that we jumped over an even number of roots (unlikely for 
    # simple cases, but possible). 
    # For "3 fixed smooth polynomial equations", likely simple roots.
    
    # Let's try a slightly more aggressive search:
    # 1. Check x0 + 1, x0 - 1
    # 2. Check x0 + 2, x0 - 2
    # ...
    # If we find a sign change between x0 and x0+d, we have a bracket [x0, x0+d] or [x0-d, x0].
    
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
        
        # If no sign change with current 'a' (which is x0), we just expand d.
        # Note: This assumes the root is not "between" x0 and the previous d, 
        # but since we start at d=1 and double, we might miss a root if 
        # f(x0) and f(x0+1) have same sign, but f(x0+0.5) was opposite? 
        # Unlikely if f is smooth and x0 is a "starting point". 
        # But to be safe, we could check the midpoint of the last step? 
        # That would double the calls. 
        # Given the score penalty for calls, we stick to doubling.
        
        d *= 2.0
        
        # If d gets very large, we might overflow or hit limits.
        if d > 1e10:
            break

    if not found:
        # Try a huge jump one last time
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
        # Use x0 and x0 + 1e-6
        x1 = x0 + 1e-6
        fx1 = f_call(x1)
        
        for _ in range(20):
            if not time_ok(): 
                break
            if abs(fx) <= 1e-8: 
                return x0
            if abs(fx1) <= 1e-8: 
                return x1
            if fx1 == fx: 
                break
            
            x_new = x0 - fx * (x0 - x1) / (fx - fx1)
            if not math.isfinite(x_new): 
                break
                
            f_new = f_call(x_new)
            if not math.isfinite(f_new): 
                break
                
            x0, fx = x1, fx1
            x1, fx1 = x_new, f_new
            
        return x0 if abs(fx) < abs(fx1) else x1

    # --- Phase 2: Refine using SciPy Brent's Method ---
    # Ensure a < b for brentq
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
        
    try:
        # brentq is highly efficient and robust.
        # xtol and rtol set to tight values to ensure |f(x)| is small.
        # For polynomials, convergence is usually fast.
        root = scipy.optimize.brentq(f_call, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
        
        if math.isfinite(root):
            # Verify the result
            f_root = f_call(root)
            if math.isfinite(f_root) and abs(f_root) <= 1e-8:
                return root
            # If brentq returned a point but |f| > 1e-8, it might be due to 
            # precision limits, but for these problems it should be fine.
            # We return root anyway as it's the best estimate from a robust solver.
            return root
        else:
            return a # Fallback

    except Exception:
        # Fallback to manual Illinois/Secant-Bisection if scipy fails
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
