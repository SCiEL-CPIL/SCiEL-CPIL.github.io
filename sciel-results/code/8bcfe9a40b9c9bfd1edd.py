"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 140.0
    
    evals = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def call_f(x):
        nonlocal evals, best_x, best_abs
        if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
            return None
        evals += 1
        try:
            val = f(x)
            val = float(val)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            return None
        if math.isfinite(val) and abs(val) < best_abs:
            best_abs = abs(val)
            best_x = x
        return val
    
    def siqi(a, fa, b, fb, c, fc, max_iters=80):
        """
        Safeguarded Inverse Quadratic Interpolation.
        Maintains a bracket [a,b] and uses a third point c for IQI.
        Falls back to Illinois/bisection when IQI is not well-behaved.
        """
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        if c > b or c < a:
            # Place c inside bracket if possible, else use midpoint
            c = 0.5 * (a + b)
            fc = call_f(c)
            if fc is None:
                return None
            if abs(fc) <= TOL:
                return c
        
        for _ in range(max_iters):
            if time.monotonic() > DEADLINE:
                return None
            
            # Check convergence
            if abs(fb) <= TOL:
                return b
            if abs(fa) <= TOL:
                return a
            if abs(fc) <= TOL:
                return c
            
            # Try IQI: interpolate 1/f as quadratic in x
            # We want the root of the quadratic through (a,1/fa), (b,1/fb), (c,1/fc)
            # But safer: use the formula for IQI directly on f values
            
            # Check if we can use IQI (need distinct points and non-zero f values)
            use_iqi = True
            if abs(fa) < 1e-300 or abs(fb) < 1e-300 or abs(fc) < 1e-300:
                use_iqi = False
            if abs(a - b) < 1e-15 * max(1.0, abs(a), abs(b)):
                use_iqi = False
            if abs(a - c) < 1e-15 * max(1.0, abs(a), abs(c)):
                use_iqi = False
            if abs(b - c) < 1e-15 * max(1.0, abs(b), abs(c)):
                use_iqi = False
            
            x_new = None
            if use_iqi:
                # IQI formula: x_new = (a*b*fc/(fa*fb) + b*c*fa/(fb*fc) + a*c*fb/(fa*fc)) / 
                #                  (a/(fa) + b/(fb) + c/(fc) - 2*(a*b/(fa*fb) + b*c/(fb*fc) + a*c/(fa*fc)))
                # Simpler standard form:
                # x_new = c + ((c-a)*(c-b)*fc) / ((a-b)*(a-c)*fa + (b-a)*(b-c)*fb + (c-a)*(c-b)*fc)
                # Let's use the Lagrange-based IQI:
                # Fit quadratic p(x) through (a,fa), (b,fb), (c,fc)
                # Find root of p(x) = 0
                # p(x) = fa * (x-b)(x-c)/((a-b)(a-c)) + fb * (x-a)(x-c)/((b-a)(b-c)) + fc * (x-a)(x-b)/((c-a)(c-b))
                # Setting p(x) = 0 and solving for x gives:
                # x = [fa*(b*c)/((a-b)*(a-c)) + fb*(a*c)/((b-a)*(b-c)) + fc*(a*b)/((c-a)*(c-b))] /
                #     [fa/((a-b)*(a-c)) + fb/((b-a)*(b-c)) + fc/((c-a)*(c-b))]
                
                denom = (a - b) * (a - c)
                denom_b = (b - a) * (b - c)
                denom_c = (c - a) * (c - b)
                
                if abs(denom) < 1e-300 or abs(denom_b) < 1e-300 or abs(denom_c) < 1e-300:
                    use_iqi = False
                else:
                    num = fa * b * c / denom + fb * a * c / denom_b + fc * a * b / denom_c
                    den = fa / denom + fb / denom_b + fc / denom_c
                    if abs(den) > 1e-300:
                        x_new = num / den
                    else:
                        use_iqi = False
            
            if not use_iqi or not math.isfinite(x_new) or not (a < x_new < b):
                # Fall back to Illinois method (modified bisection)
                if abs(fb) < abs(fa):
                    x_new = 0.5 * (a + b)
                else:
                    x_new = (a * fb - b * fa) / (fb - fa)
                    if not math.isfinite(x_new) or not (a < x_new < b):
                        x_new = 0.5 * (a + b)
            
            f_new = call_f(x_new)
            if f_new is None:
                return None
            if abs(f_new) <= TOL:
                return x_new
            
            # Update bracket: keep the subinterval that brackets the root
            if fa * f_new < 0:
                # Root is in [a, x_new]
                b, fb = x_new, f_new
            elif fb * f_new < 0:
                # Root is in [x_new, b]
                a, fa = x_new, f_new
            else:
                # No bracket change; replace the point with larger |f|
                if abs(fa) > abs(fb):
                    a, fa = x_new, f_new
                else:
                    b, fb = x_new, f_new
            
            # Update c: keep the point that gives best IQI potential
            # Use the point with smallest |f| as c for next iteration
            if abs(fa) < abs(fb) and abs(fa) < abs(fc):
                c, fc = a, fa
            elif abs(fb) < abs(fa) and abs(fb) < abs(fc):
                c, fc = b, fb
            else:
                # Keep c as is or use midpoint
                pass
            
            # Convergence check
            if abs(b - a) <= 1e-14 * max(1.0, abs(a), abs(b)):
                return 0.5 * (a + b)
        
        # Return best of the three
        if abs(fa) <= abs(fb) and abs(fa) <= abs(fc):
            return a
        elif abs(fb) <= abs(fa) and abs(fb) <= abs(fc):
            return b
        else:
            return c
    
    def illinois(a, fa, b, fb, max_iters=60):
        """Illinois method: modified bisection with geometric weighting."""
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        for _ in range(max_iters):
            if time.monotonic() > DEADLINE:
                return None
            
            if abs(fb) <= TOL:
                return b
            if abs(fa) <= TOL:
                return a
            
            # Illinois: weight toward the side with smaller |f|
            if abs(fb) < abs(fa):
                x_new = (a * fb - b * fa) / (fb - fa)
            else:
                x_new = 0.5 * (a + b)
            
            if not math.isfinite(x_new) or not (a < x_new < b):
                x_new = 0.5 * (a + b)
            
            f_new = call_f(x_new)
            if f_new is None:
                return None
            if abs(f_new) <= TOL:
                return x_new
            
            if fa * f_new < 0:
                b, fb = x_new, f_new
            elif fb * f_new < 0:
                a, fa = x_new, f_new
            else:
                if abs(fa) > abs(fb):
                    a, fa = x_new, f_new
                else:
                    b, fb = x_new, f_new
            
            if abs(b - a) <= 1e-14 * max(1.0, abs(a), abs(b)):
                return 0.5 * (a + b)
        
        if abs(fa) <= abs(fb):
            return a
        else:
            return b
    
    def find_bracket(x, fx, max_expand=30):
        """Find a bracket by expanding asymmetrically."""
        if fx is None:
            return None
        if abs(fx) <= TOL:
            return (x, x, fx, fx)
        
        base_step = 0.05 * max(1.0, abs(x))
        
        # Try expanding in both directions, alternating
        for direction in (1.0, -1.0):
            step = base_step
            cur = x
            curf = fx
            for _ in range(max_expand):
                if time.monotonic() > DEADLINE:
                    return None
                nxt = x + direction * step
                fn = call_f(nxt)
                if fn is None:
                    break
                if curf * fn < 0.0:
                    if direction > 0:
                        return (cur, nxt, curf, fn)
                    else:
                        return (nxt, cur, fn, curf)
                cur, curf = nxt, fn
                step *= 1.618  # Golden ratio for smoother expansion
        
        return None
    
    def try_solve_from_bracket(lo, hi, flo, fhi):
        """Try to solve using SIQI, then Illinois as fallback."""
        if abs(flo) <= TOL:
            return lo
        if abs(fhi) <= TOL:
            return hi
        
        # Get a third point for SIQI
        mid = 0.5 * (lo + hi)
        fm = call_f(mid)
        if fm is None:
            return None
        if abs(fm) <= TOL:
            return mid
        
        # Try SIQI first
        result = siqi(lo, flo, hi, fhi, mid, fm, max_iters=50)
        if result is not None and abs(f(result)) <= TOL if False else True:
            # Verify
            f_res = call_f(result) if result is not None else None
            if f_res is not None and abs(f_res) <= TOL:
                return result
            # SIQI returned something but not converged; try Illinois
            result2 = illinois(lo, flo, hi, fhi, max_iters=40)
            if result2 is not None:
                f_res2 = call_f(result2)
                if f_res2 is not None and abs(f_res2) <= TOL:
                    return result2
            return result
        
        # Fallback to Illinois
        result = illinois(lo, flo, hi, fhi, max_iters=60)
        if result is not None:
            f_res = call_f(result)
            if f_res is not None and abs(f_res) <= TOL:
                return result
            return result
        
        return None
    
    def multi_start(x0, n_candidates=20):
        """Multi-start with adaptive candidate generation."""
        x0 = float(x0)
        scale = max(1.0, abs(x0))
        
        # Generate diverse candidates
        candidates = [x0]
        # Log-spaced offsets
        for exp in range(-8, 5):
            s = 10.0 ** exp
            candidates.append(x0 + s * scale)
            candidates.append(x0 - s * scale)
        
        # Add some random perturbations
        rng = random.Random(12345)
        for _ in range(10):
            s = rng.uniform(0.01, 100.0) * scale
            sign = 1.0 if rng.random() < 0.5 else -1.0
            candidates.append(x0 + sign * s)
        
        # Shuffle
        rng.shuffle(candidates)
        
        brackets = []
        for x_try in candidates:
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 30:
                break
            fx = call_f(x_try)
            if fx is None:
                continue
            if abs(fx) <= TOL:
                return x_try
            bracket = find_bracket(x_try, fx, max_expand=15)
            if bracket is not None:
                lo, hi, flo, fhi = bracket
                if abs(flo) <= TOL:
                    return lo
                if abs(fhi) <= TOL:
                    return hi
                brackets.append((lo, hi, flo, fhi))
        
        # Sort by bracket width (prefer narrow brackets)
        brackets.sort(key=lambda b: abs(b[1] - b[0]))
        
        # Try each bracket
        for lo, hi, flo, fhi in brackets:
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 15:
                break
            result = try_solve_from_bracket(lo, hi, flo, fhi)
            if result is not None:
                return result
        
        return None
    
    # Phase 1: Quick secant from x0 (2 evaluations)
    x0 = float(x0)
    h = 1e-5 * max(1.0, abs(x0))
    x1 = x0 + h
    f0 = call_f(x0)
    f1 = call_f(x1)
    if f0 is not None and abs(f0) <= TOL:
        return x0
    if f1 is not None and abs(f1) <= TOL:
        return x1
    
    if f0 is not None and f1 is not None and f0 != f1:
        # Secant iterations
        for _ in range(20):
            if time.monotonic() > DEADLINE:
                break
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            if not math.isfinite(x2):
                break
            f2 = call_f(x2)
            if f2 is None:
                break
            if abs(f2) <= TOL:
                return x2
            x0, x1 = x1, x2
            f0, f1 = f1, f2
    
    # Phase 2: Multi-start with SIQI
    result = multi_start(x0)
    if result is not None:
        return result
    
    # Phase 3: If we have a good best_x, try to refine
    if best_abs < 1e-4:
        # Try secant from best_x
        h = 1e-5 * max(1.0, abs(best_x))
        x1 = best_x + h
        f0 = call_f(best_x)
        f1 = call_f(x1)
        if f0 is not None and f1 is not None and f0 != f1:
            for _ in range(30):
                if time.monotonic() > DEADLINE:
                    break
                x2 = x1 - f1 * (x1 - best_x) / (f1 - f0)
                if not math.isfinite(x2):
                    break
                f2 = call_f(x2)
                if f2 is None:
                    break
                if abs(f2) <= TOL:
                    return x2
                best_x, x1 = x1, x2
                f0, f1 = f1, f2
    
    return best_x
# EVOLVE-BLOCK-END
