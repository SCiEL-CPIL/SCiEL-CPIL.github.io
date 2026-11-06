"""Derivative-free trust-region root finder for ill-conditioned 1D problems."""
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using derivative-free methods.
    Strategy:
      1. Try a robust bracketing + bisection / Brent hybrid if possible.
      2. If no bracket is found, use a local search (Nelder-Mead-like) on |f(x)|.
      3. Fall back to secant method with safeguards.
    
    Budget: at most 500 evaluations of f.
    """
    target_tol = 1e-8
    max_evals = 500
    eval_count = [0]

    def eval_f(x):
        eval_count[0] += 1
        try:
            val = f(x)
            if not (isinstance(val, (int, float)) or (hasattr(val, '__float__'))):
                val = float(val)
            if math.isnan(val) or math.isinf(val):
                return float('inf')
            return val
        except Exception:
            return float('inf')

    def abs_f(x):
        return abs(eval_f(x))

    # --- Phase 1: Attempt to find a bracket around x0 ---
    x = float(x0)
    f0 = eval_f(x)
    if abs(f0) <= target_tol:
        return x

    # Try expanding a bracket
    bracket_found = False
    a, b = None, None
    fa, fb = None, None

    # Try small steps first, then larger
    step_sizes = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    
    for h in step_sizes:
        if eval_count[0] >= max_evals - 10:
            break
        x1 = x + h
        x2 = x - h
        f1 = eval_f(x1)
        f2 = eval_f(x2)
        
        if abs(f1) <= target_tol:
            return x1
        if abs(f2) <= target_tol:
            return x2
        
        # Check if we have a sign change between x and x1
        if f0 * f1 < 0:
            a, b = x, x1
            fa, fb = f0, f1
            bracket_found = True
            break
        # Check if we have a sign change between x and x2
        if f0 * f2 < 0:
            a, b = x2, x
            fa, fb = f2, f0
            bracket_found = True
            break
        # Check if x1 and x2 have opposite signs
        if f1 * f2 < 0:
            a, b = x1, x2
            fa, fb = f1, f2
            bracket_found = True
            break

    if bracket_found:
        # --- Phase 2: Use Brent's method or bisection on the bracket ---
        # Implement a simplified Brent's method
        def brent(a, b, fa, fb, tol=1e-14, max_iter=100):
            """Brent's root-finding method."""
            if fa * fb > 0:
                return None
            
            # Ensure |fa| < |fb|
            if abs(fa) > abs(fb):
                a, b = b, a
                fa, fb = fb, fa
            
            c = a
            fc = fa
            d = b - a
            e = d
            
            for _ in range(max_iter):
                if eval_count[0] >= max_evals - 5:
                    break
                if abs(b - a) < tol:
                    return b
                if fa * fc < 0:
                    a = c
                    fa = fc
                    d = b - a
                    e = d
                if abs(fc) < abs(fa):
                    a = b
                    b = c
                    c = a
                    fa = fb
                    fb = fc
                    fc = fa
                    d = b - a
                    e = d
                
                # Inverse quadratic interpolation
                tol1 = 2e-16 * max(1.0, abs(b))
                m1 = (fa - fb) / (b - a)
                m2 = (fc - fb) / (c - a)
                s = fc
                if abs(s) < abs(m1):
                    s = m2
                else:
                    s = m1
                if abs(s) < abs(m2):
                    s = m1
                else:
                    s = m2
                
                t = 0.5 * (a + b)
                if abs(b - t) <= max(tol1, 1e-14):
                    s = (fa - fb) / (b - a)
                
                if (a + e) <= t and t <= (b + e):
                    # Bisection
                    s = (b - a) / 2.0
                    if b - a > 0:
                        s = -s
                else:
                    # Interpolation
                    if a == c or b == c:
                        s = (b - a) / 2.0
                        if b - a > 0:
                            s = -s
                    else:
                        # Inverse quadratic
                        num = fa * fc * (fb - fa)
                        den = fa * fb * (fc - fa)
                        if den != 0:
                            s = num / den
                            # Ensure s is within bounds
                            if s > (b - a):
                                s = b - a
                            elif s < (a - b):
                                s = a - b
                            # Check if interpolation is too close to b
                            if abs(s) < 2 * abs(b - t):
                                s = 2 * (b - t)
                                if s * (b - t) < 0:
                                    s = -s
                if s == 0:
                    s = (b - a) / 2.0
                    if b - a > 0:
                        s = -s
                
                new_x = b + s
                new_f = eval_f(new_x)
                if abs(new_f) <= target_tol:
                    return new_x
                
                c = b
                fc = fb
                b = new_x
                fb = new_f
                
                if eval_count[0] >= max_evals - 2:
                    break
            
            # Return the best point
            if abs(fa) < abs(fb):
                return a
            return b

        result = brent(a, b, fa, fb)
        if result is not None and abs(eval_f(result)) <= target_tol:
            return result
        # If Brent didn't meet tolerance, try bisection as fallback
        # Actually, let's just return the best of a, b, or the last iteration
        if abs(fa) < abs(fb):
            return a
        return b

    # --- Phase 3: No bracket found. Use local minimization of |f(x)| ---
    # Nelder-Mead simplex method in 1D (essentially a golden section or adaptive search)
    
    # Start with a small simplex
    scale = max(1e-4, 1e-4 * abs(x))
    x1 = x + scale
    x2 = x - scale
    
    f_x = abs_f(x)
    f_x1 = abs_f(x1)
    f_x2 = abs_f(x2)
    
    # Keep the three best points
    points = [(x, f_x), (x1, f_x1), (x2, f_x2)]
    points.sort(key=lambda p: p[1])
    
    best_x, best_f = points[0]
    
    # Adaptive local search
    step = scale
    for i in range(min(200, max_evals - eval_count[0])):
        if best_f <= target_tol:
            break
        if eval_count[0] >= max_evals - 5:
            break
        
        # Try expanding in the best direction
        # Determine direction from the two best points
        if len(points) >= 2:
            p1, p2 = points[0], points[1]
            direction = 1.0 if p1[0] > p2[0] else -1.0
            # Actually, let's try both directions
            candidates = [
                best_x + direction * step,
                best_x - direction * step,
                best_x + step,
                best_x - step
            ]
        else:
            candidates = [best_x + step, best_x - step]
        
        best_candidate = None
        best_candidate_f = best_f
        
        for cand in candidates:
            if eval_count[0] >= max_evals - 2:
                break
            cf = abs_f(cand)
            if cf < best_candidate_f:
                best_candidate_f = cf
                best_candidate = cand
        
        if best_candidate is not None:
            best_x = best_candidate
            best_f = best_candidate_f
            # Shrink step if we made progress
            step *= 0.5
        else:
            # No improvement, try larger step
            step *= 2.0
            if step > 1e10:
                break
    
    if best_f <= target_tol:
        return best_x
    
    # --- Phase 4: Secant method with safeguards as final attempt ---
    # Use the two best points from our search
    if len(points) >= 2:
        x_a, f_a = points[0]
        x_b, f_b = points[1]
    else:
        x_a, f_a = x, f0
        x_b, f_b = x + scale, eval_f(x + scale)
    
    for i in range(min(50, max_evals - eval_count[0])):
        if eval_count[0] >= max_evals - 2:
            break
        if f_a * f_b < 0:
            # Secant step
            if abs(f_b - f_a) < 1e-30:
                break
            x_new = x_b - f_b * (x_b - x_a) / (f_b - f_a)
            # Safeguard: keep within a reasonable range
            if abs(x_new - x_b) > 10 * abs(x_b - x_a):
                x_new = x_b + 0.5 * (x_b - x_a)
            f_new = eval_f(x_new)
            if abs(f_new) <= target_tol:
                return x_new
            # Update
            x_a, f_a = x_b, f_b
            x_b, f_b = x_new, f_new
        else:
            # No sign change, try bisection-like approach
            mid = 0.5 * (x_a + x_b)
            f_mid = eval_f(mid)
            if abs(f_mid) <= target_tol:
                return mid
            if abs(f_mid) < abs(f_a):
                x_a, f_a = mid, f_mid
            else:
                x_b, f_b = mid, f_mid
    
    # Return the best point we found
    if abs(f_a) < abs(f_b):
        return x_a
    return x_b

# EVOLVE-BLOCK-END
