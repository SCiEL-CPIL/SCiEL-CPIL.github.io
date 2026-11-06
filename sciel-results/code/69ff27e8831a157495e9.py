"""Root finding for ill-conditioned equations using a hybrid approach."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root x of f(x)=0 such that |f(x)| <= 1e-8.
    Uses a hybrid strategy:
      1. Try Newton/secant with safeguards.
      2. If that fails, fall back to a robust bracketing + bisection +
         secant hybrid (similar to scipy.optimize.brentq but self-contained).
      3. Track evaluations and time; return best candidate.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    max_evals = 490  # leave a small buffer
    eval_count = [0]
    
    def _f(x):
        if time.time() > deadline:
            raise TimeoutError
        eval_count[0] += 1
        if eval_count[0] > max_evals:
            raise ValueError("too many evals")
        val = f(x)
        if not math.isfinite(val):
            return float('inf') if val > 0 else float('-inf')
        return val
    
    def _abs_f(x):
        return abs(_f(x))
    
    best_x = float(x0)
    best_val = float('inf')
    
    def _update_best(x):
        nonlocal best_x, best_val
        try:
            fx = f(x)  # don't count this as an extra eval beyond what we already did
            # Actually we should use the counted version, but we already have fx from _f
            # Let's just track via _f calls
        except:
            pass
    
    # Helper: get f(x) with counting, update best
    def _eval(x):
        nonlocal best_x, best_val
        val = _f(x)
        av = abs(val)
        if av < best_val:
            best_val = av
            best_x = x
        return val
    
    x0 = float(x0)
    _eval(x0)
    
    if best_val <= 1e-8:
        return best_x
    
    # --- Phase 1: Secant method with safeguards ---
    # Start with two points
    h = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + h
    x2 = x0 - h
    f1 = _eval(x1)
    f2 = _eval(x2)
    
    # Choose the two points with smallest |f| as initial secant pair
    if abs(f1) <= abs(f2):
        xa, fa = x1, f1
        xb, fb = x2, f2
    else:
        xa, fa = x2, f2
        xb, fb = x1, f1
    
    # Secant iterations
    for i in range(80):
        if time.time() > deadline:
            break
        if abs(fa) <= 1e-8:
            best_x = xa
            best_val = abs(fa)
            break
        if abs(fb) <= 1e-8:
            best_x = xb
            best_val = abs(fb)
            break
        
        if fa == fb:
            break
        
        # Secant step
        denom = fb - fa
        if abs(denom) < 1e-300:
            break
        xn = xb - fb * (xb - xa) / denom
        
        # Safeguard: keep xn within a reasonable range
        # Clamp to [min(xa,xb)*0.5 - 1, max(xa,xb)*0.5 + 1] roughly
        lo = min(xa, xb)
        hi = max(xa, xb)
        span = hi - lo
        if span > 0:
            if xn < lo - 2*span or xn > hi + 2*span:
                xn = 0.5 * (lo + hi)
        
        fn = _eval(xn)
        
        # Update the pair: keep the two points with smallest |f|
        if abs(fn) < abs(fa):
            xa, fa = xb, fb
            xb, fb = xn, fn
        elif abs(fn) < abs(fb):
            xb, fb = xn, fn
        else:
            # Replace the worse of xa, xb with xn only if it's better than the worse
            if abs(fn) < max(abs(fa), abs(fb)):
                if abs(fa) > abs(fb):
                    xa, fa = xn, fn
                else:
                    xb, fb = xn, fn
            else:
                # No improvement, break
                break
    
    if best_val <= 1e-8:
        return best_x
    
    # --- Phase 2: Try to find a bracket and use bisection/secant hybrid ---
    # Search for a sign change around the best point
    a = best_x
    fa = f(a)
    # Try expanding to find a sign change
    step = 1e-2 * max(1.0, abs(a))
    found_bracket = False
    for direction in [1, -1]:
        for k in range(50):
            b = a + direction * step * (2 ** k)
            try:
                fb = f(b)
            except:
                break
            eval_count[0] += 1
            if eval_count[0] > max_evals:
                break
            if not math.isfinite(fb):
                break
            if fa * fb < 0:
                found_bracket = True
                break
            if abs(fb) < abs(fa):
                a = b
                fa = fb
            if time.time() > deadline:
                break
        if found_bracket or time.time() > deadline:
            break
    
    if found_bracket:
        # Bisection + secant hybrid (Brent-like)
        # Ensure |fa| <= |fb|
        if abs(fa) > abs(fb):
            a, fa = b, fb
            b, fb = a, fa  # swap back correctly
            # Actually let's just re-assign
            a, b = b, a
            fa, fb = fb, fa
        
        # Now fa and fb have opposite signs
        for i in range(200):
            if time.time() > deadline:
                break
            if abs(fa) <= 1e-8:
                best_x = a
                best_val = abs(fa)
                break
            if abs(fb) <= 1e-8:
                best_x = b
                best_val = abs(fb)
                break
            
            # Bisection midpoint
            c = 0.5 * (a + b)
            fc = f(c)
            eval_count[0] += 1
            if eval_count[0] > max_evals:
                break
            if not math.isfinite(fc):
                break
            
            if abs(fc) < best_val:
                best_val = abs(fc)
                best_x = c
            
            if fc == 0:
                best_x = c
                best_val = 0
                break
            
            # Maintain bracket: f(a)*f(c) <= 0
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
            
            # If bracket is small enough, break
            if abs(b - a) < 1e-14:
                break
    
    if best_val <= 1e-8:
        return best_x
    
    # --- Phase 3: Final local refinement around best_x using fine secant ---
    if time.time() < deadline - 5 and eval_count[0] < max_evals - 10:
        x = best_x
        fx = f(x)
        if math.isfinite(fx):
            for i in range(30):
                if time.time() > deadline:
                    break
                if abs(fx) <= 1e-8:
                    best_x = x
                    best_val = abs(fx)
                    break
                h = 1e-6 * max(1.0, abs(x))
                fxh = f(x + h)
                eval_count[0] += 1
                if eval_count[0] > max_evals:
                    break
                if not math.isfinite(fxh):
                    break
                d = (fxh - fx) / h
                if abs(d) < 1e-300:
                    break
                x_new = x - fx / d
                if not math.isfinite(x_new):
                    break
                # Damp if step is too large
                if abs(x_new - x) > 10 * max(1.0, abs(x)):
                    x_new = x + 0.5 * max(1.0, abs(x)) * (1 if x_new > x else -1)
                fx = f(x_new)
                eval_count[0] += 1
                if eval_count[0] > max_evals:
                    break
                if not math.isfinite(fx):
                    break
                if abs(fx) < best_val:
                    best_val = abs(fx)
                    best_x = x_new
                x = x_new
    
    return best_x
# EVOLVE-BLOCK-END
