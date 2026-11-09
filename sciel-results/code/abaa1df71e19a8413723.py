"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, using a hybrid grid-scan + refined
    root-finding approach.
    
    Phase 1: Dense uniform grid scan to identify sign-change brackets.
    Phase 2: High-precision bisection/Newton within identified brackets.
    Phase 3: Fallback to secant/Newton if grid scan fails.
    """
    deadline = time.time() + 115.0  # safety margin before 120s limit
    tol = 1e-9  # target |f(x)| <= 1e-8, use tighter internal tol
    
    x = float(x0)
    best_x = x
    best_f = f(x)
    best_abs = abs(best_f) if best_f is not None and math.isfinite(best_f) else float('inf')
    
    def update_best(x_val, f_val):
        nonlocal best_x, best_f, best_abs
        if x_val is None or f_val is None:
            return
        if not math.isfinite(x_val) or not math.isfinite(f_val):
            return
        abs_f = abs(f_val)
        if abs_f < best_abs:
            best_x = x_val
            best_f = f_val
            best_abs = abs_f
        return abs_f <= tol
    
    def check_deadline():
        return time.time() >= deadline
    
    # Check if x0 is already close to a root
    if best_abs <= tol:
        return best_x
    
    # Phase 1: Dense grid scan to find sign-change brackets
    # Use a grid centered around x0 with adaptive spacing
    n_grid = 500  # number of grid points
    # Start with a moderate range, expand if needed
    range_factor = 10.0
    
    def scan_grid(x0, range_factor, n_points):
        """Scan a grid around x0 and return list of (a, b, fa, fb) brackets."""
        if check_deadline():
            return []
        
        span = range_factor * max(1.0, abs(x0))
        if span < 1e-3:
            span = 1e-3
        
        xs = np.linspace(x0 - span, x0 + span, n_points)
        brackets = []
        
        # Evaluate f on the grid
        try:
            fs = np.array([f(float(xi)) for xi in xs])
        except:
            return []
        
        # Filter out non-finite values
        valid = np.isfinite(fs)
        
        if not np.any(valid):
            return []
        
        # Find sign changes
        for i in range(len(xs) - 1):
            if check_deadline():
                break
            if valid[i] and valid[i + 1]:
                fi = float(fs[i])
                fj = float(fs[i + 1])
                xi = float(xs[i])
                xj = float(xs[i + 1])
                
                # Update best
                if abs(fi) < best_abs:
                    update_best(xi, fi)
                if abs(fj) < best_abs:
                    update_best(xj, fj)
                
                if best_abs <= tol:
                    return [(xi, xj, fi, fj)]  # Found root, return single bracket
                
                # Check for sign change
                if fi * fj < 0:
                    brackets.append((xi, xj, fi, fj))
        
        return brackets
    
    # Try scanning with different range factors
    brackets = []
    for rf in [1.0, 5.0, 10.0, 50.0, 100.0]:
        if check_deadline():
            break
        brackets = scan_grid(x0, rf, n_grid)
        if brackets:
            break
    
    # Phase 2: Refine each bracket using bisection/Newton hybrid
    def refine_bracket(a, b, fa, fb, max_iter=100):
        """Refine a bracket [a, b] where fa*fb < 0 using bisection + Newton."""
        nonlocal best_x, best_f, best_abs
        
        if check_deadline():
            return False
        
        # Ensure fa*fb < 0
        if fa * fb > 0:
            return False
        
        # Check endpoints
        if abs(fa) <= tol:
            update_best(a, fa)
            return True
        if abs(fb) <= tol:
            update_best(b, fb)
            return True
        
        # Bisection with Newton acceleration
        for _ in range(max_iter):
            if check_deadline():
                return best_abs <= tol
            
            # Try Newton step if derivative is well-behaved
            mid = (a + b) / 2.0
            fm = f(mid)
            
            if fm is None or not math.isfinite(fm):
                # Fall back to pure bisection
                fm = fa * fb / (fb - fa) * (b - a) + fa  # Linear interpolation estimate
                if not math.isfinite(fm):
                    fm = 0.0
            
            update_best(mid, fm)
            if best_abs <= tol:
                return True
            
            # Try Newton step
            h = 1e-7 * max(1.0, abs(mid))
            try:
                fmp = f(mid + h)
                if fmp is not None and math.isfinite(fmp):
                    d = (fmp - fm) / h
                    if d != 0 and math.isfinite(d):
                        newton_x = mid - fm / d
                        # Check if Newton step stays in bracket
                        if a < newton_x < b:
                            fn = f(newton_x)
                            if fn is not None and math.isfinite(fn):
                                update_best(newton_x, fn)
                                if best_abs <= tol:
                                    return True
                                # Use Newton point if it reduces interval
                                if abs(fn) < abs(fm):
                                    if fa * fn < 0:
                                        b = newton_x
                                        fb = fn
                                    else:
                                        a = newton_x
                                        fa = fn
                                    continue
            except:
                pass
            
            # Pure bisection step
            if fa * fm < 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
            
            if abs(b - a) < 1e-14 * max(1.0, abs(mid)):
                break
        
        return best_abs <= tol
    
    # Refine all found brackets
    for (a, b, fa, fb) in brackets:
        if check_deadline():
            break
        if refine_bracket(a, b, fa, fb):
            return best_x
    
    # Phase 3: Fallback to secant method if grid scan didn't find a root
    def secant_method(x0, max_iter=200):
        nonlocal best_x, best_f, best_abs
        a = float(x0)
        fa = f(a)
        if fa is None or not math.isfinite(fa):
            return False
        if abs(fa) <= tol:
            update_best(a, fa)
            return True
        update_best(a, fa)
        
        # Choose b with a small offset
        h0 = 1e-4 * max(1.0, abs(a))
        b = a + h0
        fb = f(b)
        if fb is None or not math.isfinite(fb):
            return False
        if abs(fb) <= tol:
            update_best(b, fb)
            return True
        update_best(b, fb)
        
        for _ in range(max_iter):
            if check_deadline():
                return best_abs <= tol
            if fa == fb:
                break
            c = b - fb * (b - a) / (fb - fa)
            fc = f(c)
            if fc is None or not math.isfinite(fc):
                break
            if abs(fc) <= tol:
                update_best(c, fc)
                return True
            update_best(c, fc)
            
            # Keep the two points with smallest |f|
            if abs(fc) < abs(fa):
                a, fa = b, fb
                b, fb = c, fc
            else:
                a, fa = b, fb
                b, fb = c, fc
        return best_abs <= tol
    
    # Phase 4: Newton's method with safe steps
    def newton_method(x0, max_iter=100):
        nonlocal best_x, best_f, best_abs
        x = float(x0)
        fx = f(x)
        if fx is None or not math.isfinite(fx):
            return False
        if abs(fx) <= tol:
            update_best(x, fx)
            return True
        update_best(x, fx)
        
        for _ in range(max_iter):
            if check_deadline():
                return best_abs <= tol
            h = 1e-7 * max(1.0, abs(x))
            try:
                fxp = f(x + h)
                fxm = f(x - h)
                if fxp is None or not math.isfinite(fxp) or fxm is None or not math.isfinite(fxm):
                    break
                d = (fxp - fxm) / (2 * h)
                if d == 0 or not math.isfinite(d):
                    break
                step = fx / d
                # Limit step size to prevent divergence
                max_step = 10.0 * max(1.0, abs(x))
                if abs(step) > max_step:
                    step = max_step * (1 if step > 0 else -1)
                x_new = x - step
                if not math.isfinite(x_new):
                    break
                fx_new = f(x_new)
                if fx_new is None or not math.isfinite(fx_new):
                    break
                if abs(fx_new) <= tol:
                    update_best(x_new, fx_new)
                    return True
                update_best(x_new, fx_new)
                if abs(fx_new) >= abs(fx):
                    # Step made things worse, try smaller step
                    x_new = x - step / 2
                    fx_new = f(x_new)
                    if fx_new is None or not math.isfinite(fx_new):
                        break
                    if abs(fx_new) <= tol:
                        update_best(x_new, fx_new)
                        return True
                    update_best(x_new, fx_new)
                x, fx = x_new, fx_new
            except:
                break
        return best_abs <= tol
    
    # Try fallback methods
    if not secant_method(x0):
        newton_method(x0)
    
    # Final check
    if best_abs <= tol:
        return best_x
    
    # Return best found, even if not meeting tol (better than nothing)
    return best_x
# EVOLVE-BLOCK-END
