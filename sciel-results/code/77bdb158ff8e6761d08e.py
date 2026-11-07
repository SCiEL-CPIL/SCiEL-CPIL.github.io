"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find a root of f near x0 using a hybrid bracketing + IQI approach.
    Returns x such that |f(x)| <= 1e-8, minimizing function calls.
    """
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before 120s limit
    
    def time_ok():
        return (time.time() - start_time) < DEADLINE
    
    x = float(x0)
    
    # Evaluate at x0
    fx = f(x)
    if not math.isfinite(fx):
        return x
    if abs(fx) <= 1e-8:
        return x
    
    # Try to find a bracket [a, b] with f(a)*f(b) < 0
    # Start with small steps and expand
    a = x
    fa = fx
    step = 1.0
    b = None
    fb = None
    
    # Try expanding in both directions to find a bracket
    for _ in range(30):
        if not time_ok():
            break
        b = a + step
        fb = f(b)
        if not math.isfinite(fb):
            step *= 2.0
            continue
        if fa * fb <= 0:
            break
        # Try the other direction
        b = a - step
        fb = f(b)
        if not math.isfinite(fb):
            step *= 2.0
            continue
        if fa * fb <= 0:
            break
        step *= 2.0
        if step > 1e10:
            break
    
    if b is None or fb is None or not math.isfinite(fb):
        # No bracket found; try Newton's method as fallback
        x = float(x0)
        fx = f(x)
        for _ in range(50):
            if not time_ok():
                break
            if not math.isfinite(fx):
                break
            if abs(fx) <= 1e-8:
                return x
            h = 1e-6 * max(1.0, abs(x))
            d = (f(x + h) - fx) / h
            if not math.isfinite(d) or d == 0:
                break
            x = x - fx / d
            fx = f(x)
        return x
    
    # Now we have bracket [a, b] with fa*fb <= 0
    # Use IQI with bisection safeguard
    # We need 3 points for IQI. Start with a, b, and the midpoint or x0.
    # Let's maintain 3 points: p, q, r with their f values.
    # We'll use the standard IQI approach.
    
    # Initialize with a, b, and x (if x is inside bracket, else use midpoint)
    if a <= x <= b:
        p = a
        fp = fa
        q = x
        fq = fx
        r = b
        fr = fb
    else:
        # Use midpoint as third point
        mid = (a + b) / 2.0
        fmid = f(mid)
        if not math.isfinite(fmid):
            # Fallback to bisection
            for _ in range(100):
                if not time_ok():
                    break
                mid = (a + b) / 2.0
                fmid = f(mid)
                if not math.isfinite(fmid):
                    break
                if abs(fmid) <= 1e-8:
                    return mid
                if fa * fmid <= 0:
                    b = mid
                    fb = fmid
                else:
                    a = mid
                    fa = fmid
            return (a + b) / 2.0
        
        # Determine which side the sign change is on
        if fa * fmid <= 0:
            a, fa = mid, fmid
        else:
            b, fb = mid, fmid
        
        # Now pick 3 points: a, b, and the one closer to the root
        # Use a, b, and the point with smallest |f|
        if abs(fa) < abs(fb):
            p, fp = b, fb
            q, fq = a, fa
        else:
            p, fp = a, fa
            q, fq = b, fb
        r = (p + q) / 2.0
        fr = fmid if abs(r - (a + b) / 2.0) < 1e-15 else f(r)
    
    # IQI iteration
    best_x = x
    best_fx = fx
    if abs(fx) < abs(best_fx):
        best_x = x
        best_fx = fx
    
    for _ in range(200):
        if not time_ok():
            break
        
        # Update best
        if abs(fp) < abs(best_fx):
            best_x = p
            best_fx = fp
        if abs(fq) < abs(best_fx):
            best_x = q
            best_fx = fq
        if abs(fr) < abs(best_fx):
            best_x = r
            best_fx = fr
        
        if abs(best_fx) <= 1e-8:
            return best_x
        
        # Ensure we have a bracket among our 3 points
        # Find a pair with opposite signs
        bracket = None
        for (x1, f1), (x2, f2) in [((p, fp), (q, fq)), ((q, fq), (r, fr)), ((p, fp), (r, fr))]:
            if f1 * f2 <= 0:
                if abs(f1) < abs(f2):
                    bracket = (x1, f1, x2, f2)
                else:
                    bracket = (x2, f2, x1, f1)
                break
        
        if bracket is None:
            # No bracket; try Newton step from best point
            x = best_x
            fx = best_fx
            h = 1e-6 * max(1.0, abs(x))
            d = (f(x + h) - fx) / h
            if math.isfinite(d) and d != 0:
                x_new = x - fx / d
                fx_new = f(x_new)
                if math.isfinite(fx_new) and abs(fx_new) < abs(fx):
                    p, fp = x, fx
                    q, fq = x_new, fx_new
                    r = (p + q) / 2.0
                    fr = f(r)
                    continue
            break
        
        x1, f1, x2, f2 = bracket
        # The third point is the one not in the bracket
        if x1 == p and x2 == q:
            x3, f3 = r, fr
        elif x1 == q and x2 == r:
            x3, f3 = p, fp
        else:
            x3, f3 = q, fq
        
        # IQI: fit a quadratic to (x1,f1), (x2,f2), (x3,f3) and find root
        # Using the Lagrange form for inverse quadratic interpolation
        # We want to find x such that f(x) = 0 using the quadratic through the 3 points
        
        # Standard IQI formula
        # Let's compute the IQI step
        # The quadratic interpolant in Lagrange form:
        # L(x) = f1 * (x-x2)(x-x3)/((x1-x2)(x1-x3)) + f2 * (x-x1)(x-x3)/((x2-x1)(x2-x3)) + f3 * (x-x1)(x-x2)/((x3-x1)(x3-x2))
        # Set L(x) = 0 and solve for x.
        # This is a quadratic equation: A*x^2 + B*x + C = 0
        
        # Coefficients:
        # A = f1/((x1-x2)(x1-x3)) + f2/((x2-x1)(x2-x3)) + f3/((x3-x1)(x3-x2))
        # B = f1*(x2+x3)/((x1-x2)(x1-x3)) + f2*(x1+x3)/((x2-x1)(x2-x3)) + f3*(x1+x2)/((x3-x1)(x3-x2))
        # C = f1*x2*x3/((x1-x2)(x1-x3)) + f2*x1*x3/((x2-x1)(x2-x3)) + f3*x1*x2/((x3-x1)(x3-x2))
        
        d12 = x1 - x2
        d13 = x1 - x3
        d21 = x2 - x1
        d23 = x2 - x3
        d31 = x3 - x1
        d32 = x3 - x2
        
        if d12 * d13 * d23 == 0:
            # Degenerate; use bisection
            x_new = (x1 + x2) / 2.0
        else:
            A = f1/(d12*d13) + f2/(d21*d23) + f3/(d31*d32)
            B = f1*(x2+x3)/(d12*d13) + f2*(x1+x3)/(d21*d23) + f3*(x1+x2)/(d31*d32)
            C = f1*x2*x3/(d12*d13) + f2*x1*x3/(d21*d23) + f3*x1*x2/(d31*d32)
            
            if abs(A) < 1e-30:
                # Linear case
                if abs(B) > 1e-30:
                    x_new = -C / B
                else:
                    x_new = (x1 + x2) / 2.0
            else:
                disc = B*B - 4*A*C
                if disc < 0:
                    x_new = (x1 + x2) / 2.0
                else:
                    sqrt_disc = math.sqrt(disc)
                    x_new1 = (-B + sqrt_disc) / (2*A)
                    x_new2 = (-B - sqrt_disc) / (2*A)
                    # Choose the root inside the bracket [x1, x2]
                    if (x1 <= x_new1 <= x2) or (x2 <= x_new1 <= x1):
                        x_new = x_new1
                    elif (x1 <= x_new2 <= x2) or (x2 <= x_new2 <= x1):
                        x_new = x_new2
                    else:
                        x_new = (x1 + x2) / 2.0
        
        # Safeguard: if x_new is outside the bracket, use bisection
        if not (min(x1, x2) <= x_new <= max(x1, x2)):
            x_new = (x1 + x2) / 2.0
        
        if not math.isfinite(x_new):
            x_new = (x1 + x2) / 2.0
        
        # Evaluate at x_new
        f_new = f(x_new)
        if not math.isfinite(f_new):
            # Use bisection
            x_new = (x1 + x2) / 2.0
            f_new = f(x_new)
            if not math.isfinite(f_new):
                break
        
        # Update the 3 points: drop the point with the largest |f|
        # and add the new point
        points = [(p, fp), (q, fq), (r, fr)]
        # Sort by |f|
        points.sort(key=lambda t: abs(t[1]))
        # Keep the 2 points in the bracket and the new point
        # Actually, we should keep the bracket and replace the non-bracket point
        if x1 == p and x2 == q:
            # Drop r
            p, fp = x1, f1
            q, fq = x2, f2
            r, fr = x_new, f_new
        elif x1 == q and x2 == r:
            # Drop p
            p, fp = x_new, f_new
            q, fq = x1, f1
            r, fr = x2, f2
        else:
            # Drop q
            p, fp = x1, f1
            q, fq = x_new, f_new
            r, fr = x2, f2
    
    return best_x
# EVOLVE-BLOCK-END
