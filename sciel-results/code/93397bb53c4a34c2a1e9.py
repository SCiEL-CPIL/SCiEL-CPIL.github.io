import math
import time

def find_root(f, x0):
    start_time = time.monotonic()
    deadline = start_time + 100.0  # Safety margin before 150s kill
    call_count = 0
    max_calls = 450  # Keep well under 500 to avoid penalty

    def safe_f(x):
        nonlocal call_count
        if call_count >= max_calls:
            return None
        call_count += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return None
            return val
        except Exception:
            return None

    x = float(x0)
    fx = safe_f(x)
    if fx is None:
        # If even the first call fails, we can't do much
        return x
    if abs(fx) <= 1e-8:
        return x

    # Strategy:
    # 1. Try to find a bracket [a, b] around x0.
    # 2. If found, use Illinois or Brent-like method (but we don't have scipy brentq exposed easily without extra calls? No, we can implement it).
    #    Actually, implementing a robust bracketing solver is key.
    # 3. If no bracket found, use Newton-Raphson with damping and derivative estimation.

    # Phase 1: Bracketing
    a = x - 1.0
    b = x + 1.0
    fa = safe_f(a)
    fb = safe_f(b)

    # If initial bracket fails, expand
    expansion = 1.0
    bracket_found = False
    if fa is not None and fb is not None:
        if fa * fb < 0:
            bracket_found = True
        else:
            for _ in range(30):
                if time.monotonic() > deadline:
                    break
                expansion *= 2.0
                a = x - expansion
                b = x + expansion
                fa = safe_f(a)
                fb = safe_f(b)
                if fa is None or fb is None:
                    continue
                if fa * fb < 0:
                    bracket_found = True
                    break

    if bracket_found:
        # Use Illinois Method (a variant of Bisection that converges faster)
        # or Brent's method. Let's implement a simple robust hybrid:
        # If we have a bracket, we can use Secant step if it stays in bracket, else Bisection.
        
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Initialize
        x1 = b
        f1 = fb
        x2 = a
        f2 = fa
        
        # We want the root to be in [x2, x1] where f2 and f1 have opposite signs.
        # Usually we track the "best" guess and the bracket.
        
        # Let's use a standard Illinois algorithm structure:
        # Keep [a, b] with f(a)*f(b) < 0.
        # Keep c as the previous midpoint or secant point.
        
        # Reset to a, b
        a_br = a
        b_br = b
        fa_br = fa
        fb_br = fb
        
        # Pick initial c
        # If |fa| < |fb|, c = a, else c = b? No, Illinois uses the point with smaller |f| as the "new" point?
        # Standard Illinois:
        # 1. Start with [a, b].
        # 2. c = (a+b)/2, fc = f(c).
        # 3. If f(a)*f(c) < 0, b = c, fb = fc. Else a = c, fa = fc.
        # 4. If |fb| < |fa|, swap (a,b) and (fa,fb) so that |fb| < |fa|? No, usually we keep the "newer" point as the one with smaller error?
        # Actually, Illinois is:
        #   x0 = (a+b)/2
        #   f0 = f(x0)
        #   if f(a)*f0 < 0: b = x0, fb = f0
        #   else: a = x0, fa = f0
        #   if |fa| < |fb|: a, b = b, a; fa, fb = fb, fa  (so b is the "good" point with smaller |f|)
        #   x_new = (a + b) / (fa - fb) * (fa*fb) ... no, that's not right.
        #   x_new = b - fb * (b - a) / (fb - fa)  (Secant step from a and b)
        #   if x_new is not in [a, b], use midpoint.
        #   else x_new is the next point.
        #   Update bracket.

        # Let's implement a simpler "Bisection with Secant acceleration"
        # We maintain [a, b] with sign change.
        # We also keep the last point c and fc.
        
        # Initialize c as midpoint? Or just start bisection?
        # Let's try to use the secant step if valid.
        
        # We have a, fa, b, fb.
        # Let's assume |fb| <= |fa| is not guaranteed.
        
        # Let's just use a robust loop:
        for _ in range(100):
            if time.monotonic() > deadline:
                break
            
            # Check convergence
            if abs(fa_br) <= 1e-8:
                return a_br
            if abs(fb_br) <= 1e-8:
                return b_br
            
            # If interval is small enough, return midpoint
            if abs(b_br - a_br) < 1e-10:
                return (a_br + b_br) / 2.0
            
            # Calculate Secant point
            denom = fb_br - fa_br
            if abs(denom) > 1e-15:
                x_new = b_br - fb_br * (b_br - a_br) / denom
            else:
                x_new = (a_br + b_br) / 2.0
            
            # If Secant point is outside bracket, use midpoint
            if x_new <= a_br or x_new >= b_br:
                x_new = (a_br + b_br) / 2.0
            
            f_new = safe_f(x_new)
            if f_new is None:
                # If we can't evaluate, break or fallback?
                # Return best guess
                break
            
            if abs(f_new) <= 1e-8:
                return x_new
            
            # Update bracket
            if fa_br * f_new < 0:
                # Root in [a_br, x_new]
                b_br = x_new
                fb_br = f_new
            else:
                # Root in [x_new, b_br]
                a_br = x_new
                fa_br = f_new
            
            # Illinois heuristic: if the step was small, shrink the other side more?
            # Standard Illinois:
            # If the new point is closer to the "old" good point, swap so that the "good" point is the one with smaller |f|?
            # Actually, just keeping the bracket is bisection.
            # To speed up, we can bias the midpoint towards the side with smaller |f|.
            # Let's stick to the Secant-in-Bracket (Brent-like) for now.
            
            # Optimization: If we are doing bisection (x_new was midpoint), we can just continue.
            # The Secant step usually converges superlinearly if it stays in bracket.

        # Return best estimate
        if abs(fa_br) < abs(fb_br):
            return a_br
        else:
            return b_br

    else:
        # No bracket found. Use Damped Newton-Raphson.
        # Estimate derivative.
        x_curr = x
        f_curr = fx
        
        for _ in range(50):
            if time.monotonic() > deadline:
                break
            
            # Central difference for derivative
            h = 1e-6 * max(1.0, abs(x_curr))
            x_plus = x_curr + h
            x_minus = x_curr - h
            
            f_plus = safe_f(x_plus)
            f_minus = safe_f(x_minus)
            
            if f_plus is None or f_minus is None:
                # Try forward difference
                f_plus = safe_f(x_curr + 1e-5)
                if f_plus is None:
                    break
                d = (f_plus - f_curr) / 1e-5
            else:
                d = (f_plus - f_minus) / (2 * h)
            
            if abs(d) < 1e-15:
                # Derivative is zero, try a small perturbation
                x_curr += 1e-3
                f_curr = safe_f(x_curr)
                if f_curr is None:
                    break
                continue
            
            x_new = x_curr - f_curr / d
            
            # Damping: ensure step isn't too large
            if abs(x_new - x_curr) > 10 * max(1.0, abs(x_curr)):
                x_new = x_curr - math.copysign(1.0 * max(1.0, abs(x_curr)), f_curr)
            
            f_new = safe_f(x_new)
            if f_new is None:
                break
            
            if abs(f_new) <= 1e-8:
                return x_new
            
            # If it didn't decrease |f|, backtrack?
            if abs(f_new) > abs(f_curr):
                # Try half step
                x_new = x_curr - 0.5 * (x_new - x_curr)
                f_new = safe_f(x_new)
                if f_new is None:
                    break
                if abs(f_new) > abs(f_curr):
                    # Try quarter step
                    x_new = x_curr - 0.25 * (x_new - x_curr)
                    f_new = safe_f(x_new)
                    if f_new is None:
                        break
            
            x_curr = x_new
            f_curr = f_new
        
        return x_curr
