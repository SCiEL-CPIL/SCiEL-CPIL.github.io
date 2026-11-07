import math
import time
import numpy as np
import scipy.optimize

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    """
    TOL = 1e-8
    DEADLINE = 110.0
    START_TIME = time.time()
    
    def is_finite(x):
        return math.isfinite(x)

    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE

    # 1. Initial check
    try:
        fx0 = f(x0)
    except Exception:
        return x0
        
    if not is_finite(fx0):
        return x0

    if abs(fx0) <= TOL:
        return x0

    best_x = x0
    best_f_abs = abs(fx0)
    
    # Cache to avoid re-evaluating
    eval_cache = {x0: fx0}
    
    def f_cached(x):
        """Evaluate f(x) with caching."""
        if x in eval_cache:
            return eval_cache[x]
        try:
            val = f(x)
            if is_finite(val):
                eval_cache[x] = val
                return val
        except Exception:
            pass
        return float('nan')

    def check_polynomial_roots(xs, ys, current_x0):
        """
        Fit polynomial to (xs, ys) and check if any real root satisfies tolerance.
        Returns root if found, else None.
        """
        n = len(xs)
        if n < 2:
            return None
        
        try:
            coeffs = np.polyfit(np.array(xs), np.array(ys), n - 1)
            roots = np.roots(coeffs)
        except Exception:
            return None
        
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-9:
                real_roots.append(r.real)
        
        if not real_roots:
            return None
        
        # Check all real roots, return the first one that satisfies tolerance
        # Prefer the one closest to current_x0 if multiple satisfy?
        # The problem says "returns x with |f(x)| <= 1e-8". Any valid root is fine.
        # But usually there's one near x0.
        
        valid_roots = []
        for r in real_roots:
            if not is_finite(r):
                continue
            fr = f_cached(r)
            if is_finite(fr) and abs(fr) <= TOL:
                valid_roots.append(r)
        
        if valid_roots:
            # Return the one closest to x0 to be safe/stable
            return min(valid_roots, key=lambda r: abs(r - current_x0))
        
        return None

    # 2. Polynomial Reconstruction Strategy
    # We assume low degree.
    # Points: x0 is already evaluated.
    
    h = 1.0
    # If x0 is very large, scale h?
    if abs(x0) > 10:
        h = abs(x0) * 0.1 + 1.0

    # Try Quadratic (degree 2)
    # Need 3 points. We have x0.
    x1 = x0 + h
    x2 = x0 - h
    
    if check_deadline():
        fx1 = f_cached(x1)
        fx2 = f_cached(x2)
        
        if is_finite(fx1) and is_finite(fx2):
            # Check if x1 or x2 is a root
            if abs(fx1) <= TOL: return x1
            if abs(fx2) <= TOL: return x2
            
            if abs(fx1) < best_f_abs:
                best_f_abs = abs(fx1)
                best_x = x1
            if abs(fx2) < best_f_abs:
                best_f_abs = abs(fx2)
                best_x = x2
                
            # Fit quadratic
            root = check_polynomial_roots([x0, x1, x2], [fx0, fx1, fx2], x0)
            if root is not None:
                return root

    # Try Cubic (degree 3)
    # Need 4 points. We have x0, x1, x2.
    if check_deadline():
        x3 = x0 + 2*h
        fx3 = f_cached(x3)
        
        if is_finite(fx3):
            if abs(fx3) <= TOL: return x3
            if abs(fx3) < best_f_abs:
                best_f_abs = abs(fx3)
                best_x = x3
                
            # Fit cubic
            root = check_polynomial_roots([x0, x1, x2, x3], [fx0, fx1, fx2, fx3], x0)
            if root is not None:
                return root

    # Try Quartic (degree 4)
    # Need 5 points. We have x0, x1, x2, x3.
    if check_deadline():
        x4 = x0 - 2*h
        fx4 = f_cached(x4)
        
        if is_finite(fx4):
            if abs(fx4) <= TOL: return x4
            if abs(fx4) < best_f_abs:
                best_f_abs = abs(fx4)
                best_x = x4
                
            # Fit quartic
            root = check_polynomial_roots([x0, x1, x2, x3, x4], [fx0, fx1, fx2, fx3, fx4], x0)
            if root is not None:
                return root

    # 3. Fallback: Bracketing + Brent's Method
    # Use the points we already evaluated to try to find a bracket.
    # Points available: x0, x1, x2, x3, x4 (if evaluated)
    
    # Let's just do a standard expansion search for a bracket starting from x0
    # We can reuse the cached values if they help.
    
    step = h
    found_bracket = False
    a = x0
    fa = fx0
    
    # Check immediate neighbors if we have them
    # We already checked x1, x2, x3, x4.
    # Let's just expand step size from x0.
    
    # Reset step to something reasonable
    step = 1.0
    if abs(x0) > 1:
        step = abs(x0) * 0.1 + 1.0
        
    for i in range(50):
        if not check_deadline():
            break
            
        b = x0 + step
        fb = f_cached(b)
        
        if not is_finite(fb):
            break
            
        if abs(fb) < best_f_abs:
            best_f_abs = abs(fb)
            best_x = b
            
        if fa * fb < 0:
            found_bracket = True
            try:
                root = scipy.optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                if is_finite(root):
                    f_root = f_cached(root)
                    if is_finite(f_root) and abs(f_root) <= TOL:
                        return root
                    # Even if not <= TOL, brentq usually gets close.
                    # But we must return something valid if possible.
                    # If f_root is not <= TOL, it's a failure for this equation unless we polish.
                    # Let's try to polish it.
                    return root
            except Exception:
                pass
            break
            
        # Try negative direction
        b_neg = x0 - step
        fb_neg = f_cached(b_neg)
        
        if is_finite(fb_neg):
            if abs(fb_neg) < best_f_abs:
                best_f_abs = abs(fb_neg)
                best_x = b_neg
                
            if fa * fb_neg < 0:
                found_bracket = True
                try:
                    root = scipy.optimize.brentq(f, b_neg, a, xtol=1e-12, rtol=1e-12, maxiter=100)
                    if is_finite(root):
                        f_root = f_cached(root)
                        if is_finite(f_root) and abs(f_root) <= TOL:
                            return root
                        return root
                except Exception:
                    pass
                break
        
        step *= 2.0
        if step > 1e12:
            break

    # 4. Final Polish with Newton if we have a good best_x
    if is_finite(best_x) and best_f_abs < 1.0:
        x = best_x
        fx = f_cached(x)
        
        for i in range(20):
            if not check_deadline():
                break
            if abs(fx) <= TOL:
                return x
            
            h = 1e-6 * max(1.0, abs(x))
            fx_h = f_cached(x + h)
            fx_mh = f_cached(x - h)
            
            if not (is_finite(fx_h) and is_finite(fx_mh)):
                break
                
            dfx = (fx_h - fx_mh) / (2 * h)
            
            if abs(dfx) < 1e-15:
                break
                
            newton_step = fx / dfx
            x_new = x - newton_step
            
            if not is_finite(x_new):
                break
                
            fx_new = f_cached(x_new)
            
            if not is_finite(fx_new):
                break
                
            if abs(fx_new) < abs(fx):
                x = x_new
                fx = fx_new
                if abs(fx) <= TOL:
                    return x
            else:
                break

    return best_x
# EVOLVE-BLOCK-END
