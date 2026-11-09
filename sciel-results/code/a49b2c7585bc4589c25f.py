"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f near x0 using a hybrid approach:
    1. Try Newton's method with finite-difference derivative (fast for smooth polynomials)
    2. If Newton fails or diverges, fall back to bisection on a bracketed interval
    3. Use scipy.optimize.brentq if available as another fallback
    
    Returns x such that |f(x)| <= 1e-8, or the best x found.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # 10 second safety margin
    
    x = float(x0)
    best_x = x
    best_fval = abs(f(x))
    
    # Count function calls manually
    call_count = [0]
    
    def counted_f(x):
        call_count[0] += 1
        val = f(x)
        if not math.isfinite(val):
            return val
        if abs(val) < best_fval:
            best_fval = abs(val)
            best_x = x
        return val
    
    # First, evaluate at x0
    fx0 = counted_f(x)
    if abs(fx0) <= 1e-8:
        return x
    
    # Try Newton's method with finite-difference derivative
    x_newton = x
    fx_newton = fx0
    newton_success = False
    
    for i in range(50):
        if time.time() > deadline:
            break
        if not math.isfinite(fx_newton):
            break
        
        if abs(fx_newton) <= 1e-8:
            newton_success = True
            break
        
        # Finite difference derivative
        h = 1e-7 * max(1.0, abs(x_newton))
        fx_plus = counted_f(x_newton + h)
        if not math.isfinite(fx_plus):
            break
        dfx = (fx_plus - fx_newton) / h
        
        if abs(dfx) < 1e-15:
            break
        
        step = fx_newton / dfx
        x_candidate = x_newton - step
        
        # Damping if step is too large
        if abs(step) > 10.0 * max(1.0, abs(x_newton)):
            step = math.copysign(10.0 * max(1.0, abs(x_newton)), step)
            x_candidate = x_newton - step
        
        fx_candidate = counted_f(x_candidate)
        if not math.isfinite(fx_candidate):
            break
        
        if abs(fx_candidate) < abs(fx_newton):
            x_newton = x_candidate
            fx_newton = fx_candidate
            if abs(fx_newton) <= 1e-8:
                newton_success = True
                break
        else:
            # Halve the step
            step = step / 2.0
            x_candidate = x_newton - step
            fx_candidate = counted_f(x_candidate)
            if not math.isfinite(fx_candidate):
                break
            if abs(fx_candidate) < abs(fx_newton):
                x_newton = x_candidate
                fx_newton = fx_candidate
            else:
                break
    
    if newton_success:
        return x_newton
    
    # Newton didn't converge well. Try to find a bracket and use bisection/Brent.
    # Expand interval around x0 to find a sign change
    a = x0
    fa = fx0
    b = None
    fb = None
    
    # Try expanding in both directions
    for direction in [1, -1]:
        if time.time() > deadline:
            break
        step_size = 1.0
        for _ in range(50):
            if time.time() > deadline:
                break
            b = x0 + direction * step_size
            fb = counted_f(b)
            if not math.isfinite(fb):
                step_size *= 2.0
                continue
            if fa * fb <= 0:
                break
            step_size *= 2.0
            if step_size > 1e6:
                break
        if fa * fb <= 0:
            break
    
    if b is not None and math.isfinite(fb) and fa * fb <= 0:
        # We have a bracket [a, b] with a sign change. Use bisection.
        lo, hi = a, b
        flo, fhi = fa, fb
        # Ensure flo < 0 < fhi or vice versa
        if flo > 0:
            lo, hi = hi, lo
            flo, fhi = fhi, flo
        
        for _ in range(100):
            if time.time() > deadline:
                break
            mid = (lo + hi) / 2.0
            fmid = counted_f(mid)
            if not math.isfinite(fmid):
                break
            if abs(fmid) <= 1e-8:
                return mid
            if flo * fmid <= 0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid
            if hi - lo < 1e-12:
                break
        
        # Return the midpoint or the endpoint with smaller |f|
        mid = (lo + hi) / 2.0
        fmid = counted_f(mid)
        if abs(fmid) <= abs(flo):
            if abs(fmid) <= abs(fhi):
                return mid
            else:
                return hi
        else:
            if abs(flo) <= abs(fhi):
                return lo
            else:
                return hi
    
    # No bracket found. Try a few more strategies.
    # Try scipy brentq with a wider bracket
    try:
        from scipy.optimize import brentq
        # Try to find a bracket with wider search
        for scale in [0.1, 0.5, 1, 2, 5, 10, 50, 100]:
            if time.time() > deadline:
                break
            a2 = x0 - scale
            b2 = x0 + scale
            fa2 = counted_f(a2)
            fb2 = counted_f(b2)
            if math.isfinite(fa2) and math.isfinite(fb2) and fa2 * fb2 <= 0:
                try:
                    root = brentq(counted_f, a2, b2, xtol=1e-12, rtol=1e-14, maxiter=50)
                    if math.isfinite(root) and abs(counted_f(root)) <= 1e-8:
                        return root
                except:
                    pass
    except ImportError:
        pass
    
    # Return the best x found so far
    return best_x
# EVOLVE-BLOCK-END
