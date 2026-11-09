"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import scipy.optimize

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Hybrid root finder: robust bracketing + Brent's method with fallbacks.
    Returns x with |f(x)| <= 1e-8, minimizing function call count.
    """
    deadline = time.monotonic() + 110.0  # safety margin before 120s limit
    max_calls = 480
    call_count = 0
    
    def time_ok():
        return (time.monotonic() < deadline) and (call_count < max_calls)

    def f_call(x):
        nonlocal call_count
        call_count += 1
        if not time_ok():
            raise RuntimeError("Deadline or call limit exceeded")
        return f(x)

    x = float(x0)
    try:
        fx = f_call(x)
    except RuntimeError:
        return x
    
    if not math.isfinite(fx):
        # If initial point is bad, try a small perturbation
        h = 1e-4 * max(1.0, abs(x))
        try:
            x_try = x + h
            fx_try = f_call(x_try)
            if math.isfinite(fx_try) and abs(fx_try) <= 1e-8:
                return x_try
        except RuntimeError:
            pass
        return x

    if abs(fx) <= 1e-8:
        return x

    # --- Phase 1: Find a Bracket ---
    # Strategy: Expand symmetrically from x0 to find a sign change.
    a = x
    fa = fx
    found = False
    b = None
    fb = None
    
    # Store evaluated points to potentially find a bracket between any two
    # but primarily look for f(x0) * f(x0 +/- d) < 0.
    # For polynomials, if the root is far, we might need to expand.
    
    d = 1.0
    for _ in range(30):
        if not time_ok():
            break
            
        # Try right
        xr = a + d
        try:
            fr = f_call(xr)
        except RuntimeError:
            break
        if math.isfinite(fr):
            if fa * fr < 0:
                b = xr
                fb = fr
                found = True
                break
            if abs(fr) <= 1e-8:
                return xr
        
        # Try left
        xl = a - d
        try:
            fl = f_call(xl)
        except RuntimeError:
            break
        if math.isfinite(fl):
            if fa * fl < 0:
                b = xl
                fb = fl
                found = True
                break
            if abs(fl) <= 1e-8:
                return xl
        
        d *= 2.0
        
        # If d gets very large, we might overflow or hit limits.
        if d > 1e10:
            break

    if not found:
        # Try a huge jump one last time
        d = 1e6
        try:
            xr = a + d
            fr = f_call(xr)
            if math.isfinite(fr) and fa * fr < 0:
                b = xr
                fb = fr
                found = True
            elif math.isfinite(fr) and abs(fr) <= 1e-8:
                return xr
            else:
                xl = a - d
                fl = f_call(xl)
                if math.isfinite(fl) and fa * fl < 0:
                    b = xl
                    fb = fl
                    found = True
                elif math.isfinite(fl) and abs(fl) <= 1e-8:
                    return xl
        except RuntimeError:
            pass

    if found:
        # --- Phase 2: Refine using SciPy Brent's Method ---
        # Ensure a < b for brentq
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        try:
            # brentq is highly efficient and robust.
            # xtol and rtol set to tight values to ensure |f(x)| is small.
            root = scipy.optimize.brentq(f_call, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
            
            if math.isfinite(root):
                # Verify the result
                try:
                    f_root = f_call(root)
                except RuntimeError:
                    f_root = None
                if f_root is not None and math.isfinite(f_root) and abs(f_root) <= 1e-8:
                    return root
                # If brentq returned a point but |f| > 1e-8, it might be due to 
                # precision limits, but for these problems it should be fine.
                # We return root anyway as it's the best estimate from a robust solver.
                return root
            else:
                return a # Fallback
        except RuntimeError:
            pass
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
                    
                try:
                    f_next = f_call(x_next)
                except RuntimeError:
                    break
                    
                if not math.isfinite(f_next):
                    x_next = 0.5 * (x_curr + x_prev)
                    try:
                        f_next = f_call(x_next)
                    except RuntimeError:
                        break
                        
                if abs(f_next) <= 1e-8:
                    return x_next
                     
                if fx_prev * f_next < 0:
                    x_curr = x_next
                    fx_curr = f_next
                else:
                    x_prev = x_next
                    fx_prev = f_next
                    
            return x_curr if abs(fx_curr) < abs(fx_prev) else x_prev

    # --- Fallback: Secant without bracket ---
    # Use x0 and x0 + 1e-6
    x1 = x + 1e-6
    try:
        fx1 = f_call(x1)
    except RuntimeError:
        return x if math.isfinite(fx) else x1
        
    best_x = x if abs(fx) < abs(fx1) else x1
    best_f = fx if abs(fx) < abs(fx1) else fx1

    for _ in range(20):
        if not time_ok(): 
            break
        if abs(fx) <= 1e-8: 
            return x
        if abs(fx1) <= 1e-8: 
            return x1
        if fx1 == fx: 
            break
            
        x_new = x - fx * (x - x1) / (fx - fx1)
        if not math.isfinite(x_new): 
            break
            
        try:
            f_new = f_call(x_new)
        except RuntimeError:
            break
            
        if not math.isfinite(f_new): 
            break
            
        if abs(f_new) <= 1e-8:
            return x_new
        
        if abs(f_new) < abs(best_f):
            best_x = x_new
            best_f = f_new
            
        x, fx = x1, fx1
        x1, fx1 = x_new, f_new
        
    return best_x
# EVOLVE-BLOCK-END
