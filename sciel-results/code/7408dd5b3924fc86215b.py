"""Direct approach for ill-conditioned root finding with global search and safeguarded Newton."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    max_evals = 500
    TOL = 1e-8
    eval_count = 0
    best_x = float(x0)
    best_fabs = abs(f(best_x))
    eval_count += 1

    def call_f(x):
        nonlocal eval_count, best_x, best_fabs
        if eval_count >= max_evals:
            raise Exception("max evals")
        if time.time() > deadline:
            raise Exception("deadline")
        val = f(x)
        eval_count += 1
        if not (isinstance(val, (int, float)) and math.isfinite(val)):
            val = float('nan')
        fabs = abs(val)
        if math.isfinite(fabs) and fabs < best_fabs:
            best_fabs = fabs
            best_x = x
        return val

    def check_time():
        return time.time() < deadline

    def secant_method(x0, tol=TOL, max_iter=200):
        """Secant method with careful handling of ill-conditioning."""
        x1 = float(x0)
        # Choose a small step to estimate second point
        h = 1e-4 * max(1.0, abs(x1))
        x2 = x1 + h
        
        try:
            f1 = call_f(x1)
            f2 = call_f(x2)
        except Exception:
            return False
        
        if abs(f1) <= tol:
            return True
        if abs(f2) <= tol:
            return True
        
        for i in range(max_iter):
            if not check_time():
                return False
            if abs(f2) <= tol:
                return True
            if abs(f1) <= tol:
                return True
            
            denom = f2 - f1
            if abs(denom) < 1e-30:
                # Try a different perturbation
                h2 = 1e-4 * max(1.0, abs(x2)) * (1 + 0.1 * i)
                x2_new = x2 + h2
                try:
                    f2 = call_f(x2_new)
                except Exception:
                    return False
                x2 = x2_new
                denom = f2 - f1
                if abs(denom) < 1e-30:
                    return False
            
            x_new = x2 - f2 * (x2 - x1) / denom
            if not math.isfinite(x_new):
                return False
            if abs(x_new) > 1e15:
                return False
            
            x1, f1 = x2, f2
            x2 = x_new
            try:
                f2 = call_f(x2)
            except Exception:
                return False
        
        return abs(f1) <= tol or abs(f2) <= tol

    def safeguarded_newton(x0, tol=TOL, max_iter=100):
        """
        Safeguarded Newton-Raphson with trust region constraints.
        Uses finite difference for derivative and limits step size
        to prevent divergence in ill-conditioned problems.
        """
        x = float(x0)
        # Initial trust region
        delta = 1e-2 * max(1.0, abs(x))
        
        for i in range(max_iter):
            if not check_time():
                return False
            try:
                fx = call_f(x)
            except Exception:
                return False
            if abs(fx) <= tol:
                return True
            
            # Compute derivative using central difference for better accuracy
            h = 1e-5 * max(1.0, abs(x))
            try:
                fp = call_f(x + h)
                fm = call_f(x - h)
            except Exception:
                # Fall back to forward difference
                try:
                    fp = call_f(x + h)
                    d = (fp - fx) / h
                except Exception:
                    return False
            else:
                d = (fp - fm) / (2 * h)
            
            if abs(d) < 1e-30:
                # Derivative too small; try larger step
                h2 = 1e-3 * max(1.0, abs(x))
                try:
                    fp2 = call_f(x + h2)
                    fm2 = call_f(x - h2)
                    d = (fp2 - fm2) / (2 * h2)
                except Exception:
                    try:
                        fp2 = call_f(x + h2)
                        d = (fp2 - fx) / h2
                    except Exception:
                        return False
                if abs(d) < 1e-30:
                    return False
            
            # Newton step
            step = fx / d
            
            # Trust region constraint
            if abs(step) > delta:
                step = math.copysign(delta, step)
            
            x_new = x - step
            if not math.isfinite(x_new):
                return False
            if abs(x_new) > 1e15:
                return False
            
            # Check if step improved things
            try:
                fx_new = call_f(x_new)
            except Exception:
                # Shrink trust region and retry
                delta *= 0.5
                continue
            
            if abs(fx_new) < abs(fx):
                # Accept step, possibly expand trust region
                x = x_new
                fx = fx_new
                delta = min(2.0 * delta, 10.0 * max(1.0, abs(x)))
            else:
                # Reject step, shrink trust region
                delta *= 0.5
                if delta < 1e-15 * max(1.0, abs(x)):
                    break
        
        try:
            fx = call_f(x)
        except Exception:
            return False
        return abs(fx) <= tol

    def bisection(x0, tol=TOL, max_iter=200):
        """Bisection with bracket search."""
        x = float(x0)
        try:
            fx = call_f(x)
        except Exception:
            return False
        if abs(fx) <= tol:
            return True
        
        # Search for a bracket
        step = 1e-3 * max(1.0, abs(x))
        a, b = None, None
        fa, fb = None, None
        
        # Search right
        for i in range(40):
            x_test = x + step * (2 ** i)
            try:
                f_test = call_f(x_test)
            except Exception:
                break
            if not math.isfinite(f_test):
                break
            if fa is None:
                a, fa = x, fx
            if fa * f_test < 0:
                b, fb = x_test, f_test
                break
        
        # Search left if not found
        if b is None:
            a, b = None, None
            fa, fb = None, None
            for i in range(40):
                x_test = x - step * (2 ** i)
                try:
                    f_test = call_f(x_test)
                except Exception:
                    break
                if not math.isfinite(f_test):
                    break
                if fb is None:
                    b, fb = x, fx
                if fb * f_test < 0:
                    a, fa = x_test, f_test
                    break
        
        if a is None or b is None:
            return False
        
        # Bisection
        for i in range(max_iter):
            if not check_time():
                return False
            mid = (a + b) / 2
            try:
                fm = call_f(mid)
            except Exception:
                return False
            if abs(fm) <= tol:
                return True
            if fa * fm <= 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            if abs(b - a) < 1e-15 * max(1.0, abs(mid)):
                break
        
        return min(abs(fa), abs(fb)) <= tol

    def global_search(x0, tol=TOL):
        """
        Global search to locate the interval minimizing |f(x)| over
        a geometrically expanding domain around x0.
        Returns the best point found.
        """
        x = float(x0)
        scale = 1e-2 * max(1.0, abs(x))
        best_local_x = x
        best_local_fabs = best_fabs
        
        # Evaluate at geometrically spaced points in both directions
        for direction in [1, -1]:
            for i in range(30):
                if not check_time():
                    break
                x_test = x + direction * scale * (2 ** i)
                if abs(x_test) > 1e15:
                    break
                try:
                    f_test = call_f(x_test)
                except Exception:
                    break
                if not math.isfinite(f_test):
                    break
                fabs_test = abs(f_test)
                if fabs_test < best_local_fabs:
                    best_local_fabs = fabs_test
                    best_local_x = x_test
                if fabs_test <= tol:
                    best_x = x_test
                    best_fabs = fabs_test
                    return True
        
        # Also try intermediate points between the best and x0
        if best_local_x != x:
            # Try a few points between x and best_local_x
            for frac in [0.25, 0.5, 0.75]:
                x_mid = x + frac * (best_local_x - x)
                try:
                    f_mid = call_f(x_mid)
                except Exception:
                    continue
                if not math.isfinite(f_mid):
                    continue
                fabs_mid = abs(f_mid)
                if fabs_mid < best_local_fabs:
                    best_local_fabs = fabs_mid
                    best_local_x = x_mid
                if fabs_mid <= tol:
                    best_x = x_mid
                    best_fabs = fabs_mid
                    return True
        
        # Update global best
        if best_local_fabs < best_fabs:
            best_fabs = best_local_fabs
            best_x = best_local_x
        
        return best_fabs <= tol

    def polish_pass(x_start, tol=TOL, max_iter=50):
        """
        High-precision polish using secant method starting from a good initial guess.
        Uses tighter tolerance and more careful step sizing.
        """
        x1 = float(x_start)
        h = 1e-8 * max(1.0, abs(x1))
        x2 = x1 + h
        
        try:
            f1 = call_f(x1)
            f2 = call_f(x2)
        except Exception:
            return False
        
        if abs(f1) <= tol:
            return True
        if abs(f2) <= tol:
            return True
        
        for i in range(max_iter):
            if not check_time():
                return False
            if abs(f2) <= tol:
                return True
            if abs(f1) <= tol:
                return True
            
            denom = f2 - f1
            if abs(denom) < 1e-30:
                h2 = 1e-8 * max(1.0, abs(x2)) * (1 + 0.01 * i)
                x2_new = x2 + h2
                try:
                    f2 = call_f(x2_new)
                except Exception:
                    return False
                x2 = x2_new
                denom = f2 - f1
                if abs(denom) < 1e-30:
                    return False
            
            x_new = x2 - f2 * (x2 - x1) / denom
            if not math.isfinite(x_new):
                return False
            if abs(x_new) > 1e15:
                return False
            
            x1, f1 = x2, f2
            x2 = x_new
            try:
                f2 = call_f(x2)
            except Exception:
                return False
        
        return abs(f1) <= tol or abs(f2) <= tol

    def hybrid_ill_conditioned(x0, tol=TOL):
        """
        Direct approach for ill-conditioned problems:
        1. First try secant (most efficient for smooth functions)
        2. If that fails, try safeguarded Newton with trust region
        3. If that fails, try bisection with wider search
        4. If that fails, try global search to find better starting point
        5. After global search, retry secant and safeguarded Newton from best point
        6. Final polish pass on best point found
        """
        # Strategy 1: Secant method
        if secant_method(x0):
            return True
        
        # Strategy 2: Safeguarded Newton with trust region
        if safeguarded_newton(x0):
            return True
        
        # Strategy 3: Bisection
        if bisection(x0):
            return True
        
        # Strategy 4: Global search to find better starting point
        if global_search(x0):
            return True
        
        # Strategy 5: Retry secant from the best point found by global search
        if best_fabs < TOL * 1000 and best_fabs > 0:
            if secant_method(best_x):
                return True
        
        # Strategy 6: Retry safeguarded Newton from the best point
        if best_fabs < TOL * 1000 and best_fabs > 0:
            if safeguarded_newton(best_x):
                return True
        
        # Strategy 7: Polish pass on best point found so far
        if best_fabs < TOL * 100 and best_fabs > 0:
            if polish_pass(best_x):
                return True
        
        return False

    # Try the hybrid approach
    hybrid_ill_conditioned(x0)
    
    # Final polish: if we're close but not quite there, do one more polish pass
    if best_fabs < TOL * 1000 and best_fabs > TOL:
        polish_pass(best_x)
    
    return best_x
# EVOLVE-BLOCK-END
