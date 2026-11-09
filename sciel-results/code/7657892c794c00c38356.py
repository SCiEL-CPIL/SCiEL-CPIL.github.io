"""Root finding for ill-conditioned equations: Robust Bracketing + Brent-like Refinement."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin below 120s limit
    MAX_EVALS = 480   # Safety margin below 500 limit

    # State tracking
    eval_count = 0
    best_x = float(x0)
    best_f = None
    best_abs = float('inf')
    
    def check_budget():
        nonlocal eval_count
        if eval_count > MAX_EVALS or (time.time() - start_time) > DEADLINE:
            return True
        return False

    def safe_eval(x):
        nonlocal eval_count, best_x, best_f, best_abs
        if check_budget():
            return None
        try:
            val = f(x)
            if not math.isfinite(val):
                return None
            eval_count += 1
            abs_val = abs(val)
            if abs_val < best_abs:
                best_f = val
                best_x = x
                best_abs = abs_val
            return val
        except Exception:
            return None

    # 1. Initial Check
    best_f = safe_eval(best_x)
    if best_f is not None and best_abs <= 1e-8:
        return best_x

    def polish(x_start, max_iter=50):
        """High-precision polish using secant/Newton with tight tolerance."""
        nonlocal eval_count, best_x, best_f, best_abs
        if check_budget():
            return None
        
        x = float(x_start)
        fx = safe_eval(x)
        if fx is None:
            return None
        
        if abs(fx) <= 1e-8:
            return x
        
        # Get a second point for secant
        h = 1e-8 * max(1.0, abs(x))
        x2 = x + h
        fx2 = safe_eval(x2)
        if fx2 is None:
            return None
        
        x1, f1 = x, fx
        x0_, f0_ = x2, fx2
        
        for _ in range(max_iter):
            if check_budget():
                break
            if abs(f1) <= 1e-8:
                return x1
            if abs(f0_) <= 1e-8:
                return x0_
            
            # Try Newton if we have a good derivative estimate
            if f1 != f0_ and abs(f1 - f0_) > 1e-15:
                x_new = x1 - f1 * (x1 - x0_) / (f1 - f0_)
            else:
                # Fallback: small step
                x_new = x1 + 1e-10 * max(1.0, abs(x1))
            
            # Safeguard
            max_jump = max(1e-6, 0.1 * abs(x1))
            if abs(x_new - x1) > max_jump:
                x_new = x1 + math.copysign(max_jump, x_new - x1)
            
            f_new = safe_eval(x_new)
            if f_new is None:
                break
            
            x0_, f0_ = x1, f1
            x1, f1 = x_new, f_new
            
            if abs(x1 - x0_) < 1e-15 * max(1.0, abs(x1)):
                break
        
        return x1 if abs(f1) <= 1e-8 else (x1 if abs(f1) < best_abs else None)

    def brent_like_solve(a, b, fa, fb, max_iter=100):
        """
        Robust root finding in [a, b] using a hybrid of Bisection and Inverse Quadratic Interpolation.
        Assumes fa and fb have opposite signs.
        """
        nonlocal eval_count, best_x, best_f, best_abs
        
        if fa * fb > 0:
            return None

        # Ensure |fa| <= |fb| initially for stability
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        c = a
        fc = fa
        d = b - a
        e = d

        for _ in range(max_iter):
            if check_budget():
                break
                
            # Check for convergence
            if abs(fb) <= 1e-8:
                best_x = b
                best_f = fb
                best_abs = abs(fb)
                return b
            if abs(fa) <= 1e-8:
                best_x = a
                best_f = fa
                best_abs = abs(fa)
                return a
            if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
                # Return the one with smaller residual
                if abs(fa) < abs(fb):
                    best_x = a
                    best_f = fa
                    best_abs = abs(fa)
                    return a
                else:
                    best_x = b
                    best_f = fb
                    best_abs = abs(fb)
                    return b

            # Inverse Quadratic Interpolation
            mflag = True
            x = None
            if abs(e) > 1e-12 * max(1.0, abs(a), abs(b)) and abs(fc) > abs(fb):
                # Use IQI
                try:
                    # Compute IQI point
                    # x = b - (b-a) * (b-c) * (fb-fa) / (2 * (fa-fb) * (b-a) + (fa-fc) * (b-c) - (fb-fc) * (a-c))
                    # Simplified form for stability:
                    # Let's use the standard formula
                    num1 = (b - a) * (fb - fa)
                    num2 = (b - c) * (fb - fc)
                    num3 = (a - c) * (fa - fc)
                    den = 2.0 * (fb - fa) * num1 + (fb - fc) * num2 - (fa - fc) * num3
                    if abs(den) > 1e-300:
                        x = b - num1 * num2 / den
                        # Check if x is within bounds
                        if (min(a, b) < x < max(a, b)):
                            mflag = False
                        else:
                            mflag = True
                    else:
                        mflag = True
                except Exception:
                    mflag = True
            
            if mflag:
                # Bisection
                x = (a + b) / 2.0
                d = b - a
                e = d
            else:
                # Secant-like step (IQI)
                e = d
                d = b - a

            # Evaluate new point
            fx = safe_eval(x)
            if fx is None:
                break

            # Update best
            if abs(fx) < best_abs:
                best_x = x
                best_f = fx
                best_abs = abs(fx)

            # Update bracket
            if fa * fx < 0:
                b = x
                fb = fx
            else:
                a = x
                fa = fx
                c = b
                fc = fb

            # Ensure |fa| <= |fb|
            if abs(fa) > abs(fb):
                a, b = b, a
                fa, fb = fb, fa

        # Return best found
        if abs(fa) < abs(fb):
            best_x = a
            best_f = fa
            best_abs = abs(fa)
            return a
        else:
            best_x = b
            best_f = fb
            best_abs = abs(fb)
            return b

    def find_bracket(x_start, max_steps=50):
        """
        Exponentially expand around x_start to find a sign change.
        Returns (a, b, fa, fb) or None.
        """
        nonlocal eval_count, best_x, best_f, best_abs
        
        h = 1e-4 * max(1.0, abs(x_start))
        a = x_start
        b = x_start + h
        
        fa = safe_eval(a)
        if fa is None:
            return None
        if abs(fa) <= 1e-8:
            return (a, a, fa, fa) # Trivial root
        
        fb = safe_eval(b)
        if fb is None:
            return None
        if abs(fb) <= 1e-8:
            return (b, b, fb, fb)

        # Check for sign change
        if fa * fb < 0:
            return (a, b, fa, fb)

        # Expand
        for _ in range(max_steps):
            if check_budget():
                break
            
            # Double the step size
            h *= 2.0
            b = a + h
            fb = safe_eval(b)
            if fb is None:
                # Try other direction
                b = a - h
                fb = safe_eval(b)
                if fb is None:
                    break
            
            if fb is None:
                continue

            if abs(fb) <= 1e-8:
                return (b, b, fb, fb)

            if fa * fb < 0:
                return (a, b, fa, fb)
            
            # If we moved too far, reset anchor to b if it's getting better? 
            # Or just keep expanding from original a? 
            # Expanding from original a is safer for global search but might miss local mins if f is non-monotonic.
            # For ill-conditioned, often monotonic near root. Let's stick to expanding from a.
            # However, if fa is huge and fb is small but same sign, we might miss a local min.
            # Let's also try expanding from b in the other direction if fa*fb > 0 and |fb| < |fa|
            if abs(fb) < abs(fa):
                # Shift anchor
                a = b
                fa = fb
                # Continue expanding in same direction
                b = a + h
                fb = safe_eval(b)
                if fb is None:
                    continue
                if fa * fb < 0:
                    return (a, b, fa, fb)
                if abs(fb) <= 1e-8:
                    return (b, b, fb, fb)

        return None

    # Main Strategy:
    # 1. Try to find a bracket around x0
    bracket = find_bracket(x0)
    if bracket:
        a, b, fa, fb = bracket
        if a == b:
            return a
        result = brent_like_solve(a, b, fa, fb)
        if best_abs <= 1e-8:
            return best_x

    # 2. If no bracket found, try perturbed starts
    perturbations = [1e-2, -1e-2, 1e-3, -1e-3, 1e-1, -1e-1, 0.5, -0.5, 1.0, -1.0]
    for p in perturbations:
        if check_budget():
            break
        x_pert = x0 + p
        bracket = find_bracket(x_pert, max_steps=30)
        if bracket:
            a, b, fa, fb = bracket
            if a == b:
                return a
            result = brent_like_solve(a, b, fa, fb)
            if best_abs <= 1e-8:
                return best_x

    # 3. Global sampling on log scale
    radii = [10.0, 100.0, 1000.0, 10000.0]
    for r in radii:
        if check_budget():
            break
        samples = [x0 - r, x0 - r/2, x0 + r/2, x0 + r]
        for x_s in samples:
            if check_budget():
                break
            fs = safe_eval(x_s)
            if fs is None:
                continue
            if abs(fs) <= 1e-8:
                return x_s
            
            # Try to bracket from this point
            bracket = find_bracket(x_s, max_steps=20)
            if bracket:
                a, b, fa, fb = bracket
                if a == b:
                    return a
                result = brent_like_solve(a, b, fa, fb)
                if best_abs <= 1e-8:
                    return best_x

    # 4. Final polish if close
    if best_abs < 1e-4:
        result = polish(best_x, max_iter=50)
        if result is not None and best_abs <= 1e-8:
            return best_x

    return best_x

# EVOLVE-BLOCK-END
