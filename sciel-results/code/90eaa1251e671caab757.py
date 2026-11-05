"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 145.0
    
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
    
    def iqi(x1, f1, x2, f2, x3, f3, max_iters=60):
        """Inverse quadratic interpolation with bracketing fallback."""
        for _ in range(max_iters):
            if time.monotonic() > DEADLINE:
                return None
            # Ensure we have a bracket: two points with opposite signs
            # Find the bracket among the three points
            bracket = None
            if f1 * f2 < 0:
                bracket = (x1, f1, x2, f2)
            elif f2 * f3 < 0:
                bracket = (x2, f2, x3, f3)
            elif f1 * f3 < 0:
                bracket = (x1, f1, x3, f3)
            
            if bracket is None:
                # No bracket - try to find one or bail
                return None
            
            a, fa, b, fb = bracket
            # Inverse quadratic interpolation
            # Fit parabola through (a,fa), (b,fb), and the third point
            c, fc = (x3, f3) if (a != x3 or fa != f3) else (x1, f1) if (a != x1 or fa != f1) else (x2, f2)
            # Actually, let's use the standard IQI formula
            # We need three distinct points
            pts = [(x1, f1), (x2, f2), (x3, f3)]
            # Remove duplicates
            unique_pts = []
            for px, pf in pts:
                if not any(abs(px - ux) < 1e-15 and abs(pf - uf) < 1e-15 for ux, uf in unique_pts):
                    unique_pts.append((px, pf))
            if len(unique_pts) < 3:
                # Fall back to bisection
                mid = 0.5 * (a + b)
                fm = call_f(mid)
                if fm is None:
                    return None
                if abs(fm) <= TOL:
                    return mid
                if fa * fm < 0:
                    b, fb = mid, fm
                else:
                    a, fa = mid, fm
                if abs(b - a) <= 1e-14 * max(1.0, abs(mid)):
                    return mid
                x1, f1 = a, fa
                x2, f2 = b, fb
                x3, f3 = mid, fm
                continue
            
            (p1, fp1), (p2, fp2), (p3, fp3) = unique_pts
            
            # IQI formula: x_new = x3 + ((x3-x2)*((x3-x1)*fp3 - (x3-x2)*fp2) - (x3-x1)*((x3-x2)*fp3 - (x3-x1)*fp1)) / ...
            # Simpler: use Lagrange interpolation
            num1 = (p3 - p2) * (p3 - p1) * fp3
            num2 = (p3 - p1) * (p3 - p2) * fp2
            num3 = (p3 - p2) * (p3 - p1) * fp1
            den = (p3 - p2) * (p3 - p1) * fp3 + (p3 - p1) * (p3 - p2) * fp2 + (p3 - p2) * (p3 - p1) * fp1
            
            # Actually, let's use a cleaner IQI implementation
            # Standard IQI: given three points, interpolate 1/f vs x
            x1, f1v = p1, fp1
            x2, f2v = p2, fp2
            x3, f3v = p3, fp3
            
            # Check for division by zero
            if abs(f1v) < 1e-300 or abs(f2v) < 1e-300 or abs(f3v) < 1e-300:
                # Use bisection
                mid = 0.5 * (a + b)
                fm = call_f(mid)
                if fm is None:
                    return None
                if abs(fm) <= TOL:
                    return mid
                if fa * fm < 0:
                    b, fb = mid, fm
                else:
                    a, fa = mid, fm
                if abs(b - a) <= 1e-14 * max(1.0, abs(mid)):
                    return mid
                continue
            
            # IQI: x_new = x3 + ((x3-x2)*(x3-x1)*f3 - (x3-x1)*(x3-x2)*f2 + (x3-x2)*(x3-x1)*f1) / 
            #   ((f3-f2)*(f3-f1) - (f3-f1)*(f3-f2) + (f2-f1)*(f3-f1)) ... this is getting complex
            # Let's use a simpler approach: secant-like with three points
            
            # Use the point closest to the bracket midpoint as the new estimate
            # Or just use secant on the bracket
            if abs(fb) > abs(fa):
                x_new = a - fa * (b - a) / (fb - fa)
            else:
                x_new = b - fb * (a - b) / (fa - fb)
            
            if not math.isfinite(x_new) or not (a < x_new < b):
                x_new = 0.5 * (a + b)
            
            f_new = call_f(x_new)
            if f_new is None:
                return None
            if abs(f_new) <= TOL:
                return x_new
            
            # Update points: keep the two that form a bracket and the new point
            if fa * f_new < 0:
                # New bracket is (a, x_new)
                x2, f2 = x3, f3  # keep third point
                x3, f3 = x_new, f_new
                # But we need to maintain order... let's just track the bracket
                # Actually, let's simplify: always use the bracket and one extra point
            elif fb * f_new < 0:
                # New bracket is (b, x_new)
                x1, f1 = x3, f3
                x3, f3 = x_new, f_new
            else:
                # No new bracket, replace the point with larger |f|
                if abs(fa) > abs(fb):
                    x1, f1 = x_new, f_new
                else:
                    x2, f2 = x_new, f_new
            
            # Update bracket variables
            if fa * fb > 0:
                # Lost bracket, try to recover
                return None
            
            if abs(b - a) <= 1e-14 * max(1.0, abs(x_new)):
                return x_new
        return None
    
    def find_bracket_multi(x, fx, max_expand=40):
        """Find a bracket by expanding in both directions."""
        if fx is None:
            return None
        if abs(fx) <= TOL:
            return (x, x, fx, fx)
        
        base_step = 0.1 * max(1.0, abs(x))
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
                step *= 2.0
        return None
    
    def multi_start_search(x0, n_starts=10):
        """Try multiple starting points to find brackets."""
        x0 = float(x0)
        scale = max(1.0, abs(x0))
        
        # Generate candidate starting points
        candidates = [x0]
        for s in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            candidates.append(x0 + s * scale)
            candidates.append(x0 - s * scale)
        
        # Shuffle to diversify
        import random
        random.seed(42)
        random.shuffle(candidates)
        
        brackets = []
        for x_try in candidates:
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 20:
                break
            fx = call_f(x_try)
            if fx is None:
                continue
            if abs(fx) <= TOL:
                return x_try
            bracket = find_bracket_multi(x_try, fx, max_expand=20)
            if bracket is not None:
                lo, hi, flo, fhi = bracket
                if abs(flo) <= TOL:
                    return lo
                if abs(fhi) <= TOL:
                    return hi
                brackets.append(bracket)
        
        # Sort brackets by width (prefer narrow)
        brackets.sort(key=lambda b: abs(b[1] - b[0]))
        
        # Try IQI on each bracket
        for lo, hi, flo, fhi in brackets:
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 10:
                break
            # Initialize three points for IQI
            mid = 0.5 * (lo + hi)
            fm = call_f(mid)
            if fm is None:
                continue
            if abs(fm) <= TOL:
                return mid
            result = iqi(lo, flo, hi, fhi, mid, fm)
            if result is not None:
                return result
        
        return None
    
    def secant(x0, max_iters=30):
        """Secant method with careful handling."""
        x0 = float(x0)
        h = 1e-5 * max(1.0, abs(x0))
        x1 = x0 + h
        f0 = call_f(x0)
        f1 = call_f(x1)
        if f0 is None or f1 is None:
            return None
        if abs(f0) <= TOL:
            return x0
        if abs(f1) <= TOL:
            return x1
        for _ in range(max_iters):
            if time.monotonic() > DEADLINE:
                return None
            if f1 == f0:
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
        return None
    
    # Phase 1: Quick secant from x0
    result = secant(x0, max_iters=15)
    if result is not None:
        return result
    
    # Phase 2: Multi-start bracketing + IQI
    result = multi_start_search(x0)
    if result is not None:
        return result
    
    # Phase 3: If we have a good best_x, try secant from there
    if best_abs < 1e-4:
        result = secant(best_x, max_iters=20)
        if result is not None:
            return result
    
    # Return best found
    return best_x
# EVOLVE-BLOCK-END
