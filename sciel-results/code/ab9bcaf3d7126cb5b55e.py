"""
Root finder using Brent's Method (Inverse Quadratic Interpolation with Bisection safeguard).
Strategy:
1. Adaptive geometric expansion to find a bracket [a, b] containing a root.
2. Maintain a 3-point history (a, b, c) with |f(c)| < |f(b)|.
3. At each step, attempt Inverse Quadratic Interpolation (IQI).
4. If IQI step is invalid (outside bracket, too close to boundary, or denominator too small),
   fall back to Secant method.
5. If Secant step is invalid, fall back to Bisection.
6. This ensures superlinear convergence when possible and linear convergence otherwise,
   while strictly bounding the number of function calls.
"""

# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    try:
        x0 = float(x0)
    except (TypeError, ValueError):
        return 0.0
        
    if not math.isfinite(x0):
        return 0.0

    # 1. Evaluate at x0
    try:
        fx = f(x0)
    except Exception:
        return 0.0
        
    if not math.isfinite(fx):
        return 0.0
        
    if abs(fx) <= 1e-8:
        return x0

    # 2. Find a bracket [a, b] such that f(a)*f(b) <= 0
    # Use geometric expansion with alternating signs to handle roots on either side.
    a = x0
    fa = fx
    b = None
    fb = None
    
    # Initial step size. Since polynomials are smooth, a small step is often enough.
    # If x0 is far, we expand.
    step = 1e-2
    # Max expansion steps to avoid infinite loop if no root exists nearby (though task implies existence)
    max_expansions = 50 
    
    for i in range(max_expansions):
        if time.time() > deadline:
            return a if abs(fa) <= 1e-8 else x0
            
        # Try positive direction
        b_pos = a + step
        if math.isfinite(b_pos):
            try:
                fb_pos = f(b_pos)
            except Exception:
                fb_pos = None
            if fb_pos is not None and math.isfinite(fb_pos):
                if abs(fb_pos) <= 1e-8:
                    return b_pos
                if fa * fb_pos <= 0:
                    b = b_pos
                    fb = fb_pos
                    break
        
        # Try negative direction
        b_neg = a - step
        if math.isfinite(b_neg):
            try:
                fb_neg = f(b_neg)
            except Exception:
                fb_neg = None
            if fb_neg is not None and math.isfinite(fb_neg):
                if abs(fb_neg) <= 1e-8:
                    return b_neg
                if fa * fb_neg <= 0:
                    b = b_neg
                    fb = fb_neg
                    break
        
        step *= 2.0
        if step > 1e10:
            break

    if b is None:
        # Fallback: If no bracket found, return best effort (x0)
        # This should rarely happen for the given task constraints.
        return x0

    # 3. Brent's Method Iteration
    # State: a, b (bracket), c (previous best point)
    # Invariant: |f(c)| < |f(b)| and f(a)*f(b) <= 0
    # We start with c = a (since |fa| might not be < |fb|, we need to check/swap)
    
    # Ensure |fa| <= |fb| so we can set c=a, b=b initially?
    # Actually, standard Brent starts with a, b such that |fa| <= |fb|.
    # Let's normalize.
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    
    c = a
    fc = fa
    d = b - a  # Previous step size
    e = d      # Step size for bisection
    
    max_iter = 100
    for i in range(max_iter):
        if time.time() > deadline:
            # Return the point with smallest |f|
            if abs(fc) < abs(fb):
                return c
            else:
                return b

        # Check for convergence at any point
        if abs(fc) <= 1e-8:
            return c
        if abs(fb) <= 1e-8:
            return b

        # Ensure |fa| <= |fb| is NOT strictly required for the logic below, 
        # but the "c" point must have smaller |f| than "b".
        # We maintain: |fc| < |fb|
        
        # If |fa| < |fc|, then a is better than c. Update c to a.
        # Wait, the standard algorithm maintains a, b as bracket and c as best.
        # Let's stick to the standard Brent structure:
        # a, b are bracket. c is the previous b (or a) with smaller |f|.
        # Actually, let's just track the best point found so far in the bracket.
        
        # Standard Brent logic:
        # 1. If |fa| < |fb|, swap a and b? No, usually we keep a, b fixed as bracket endpoints
        #    but track which one is "better" for the IQI history.
        
        # Let's use a simpler, robust variation:
        # Keep (a, fa), (b, fb) as the current bracket.
        # Keep (c, fc) as the previous iterate (best guess so far).
        
        # Update c if a or b is better than current c
        if abs(fa) < abs(fc):
            c, fc = a, fa
        if abs(fb) < abs(fc):
            c, fc = b, fb
            
        if abs(fc) <= 1e-8:
            return c

        # Try IQI
        # Points: (a, fa), (b, fb), (c, fc)
        # Note: If c == a or c == b, IQI is not applicable (degenerate), use Secant.
        
        use_iqi = False
        s = None
        
        # Check if we have 3 distinct points
        if not (a == b or a == c or b == c):
            # IQI Formula:
            # s = c - fc * ( (a-c)*fa^2 / ((fa-fb)*(fa-fc)) - (b-c)*fb^2 / ((fb-fa)*(fb-fc)) )
            # Denominators must not be zero.
            den1 = (fa - fb) * (fa - fc)
            den2 = (fb - fa) * (fb - fc)
            
            if abs(den1) > 1e-15 and abs(den2) > 1e-15:
                try:
                    term1 = (a - c) * (fa * fa) / den1
                    term2 = (b - c) * (fb * fb) / den2
                    s = c - fc * (term1 - term2)
                    
                    # Safeguards:
                    # 1. s must be inside bracket (a, b)
                    min_ab = min(a, b)
                    max_ab = max(a, b)
                    
                    if min_ab < s < max_ab:
                        # 2. s should not be too close to boundaries (to avoid numerical issues)
                        # Standard check: (s - a) < 0.5 * (b - a) is not the right check.
                        # Check: (s - c) < 0.5 * (b - a) ? No.
                        # Check: (s - a) < 0.5 * (b - c) and (s - b) < 0.5 * (a - c)?
                        # Let's use the standard Brent check:
                        # (s - c) < 0.5 * (b - a) is not quite right.
                        # The condition is usually:
                        # (s - a) < 0.5 * (b - a) AND (s - b) < 0.5 * (a - b) ??
                        
                        # Let's use:
                        # (s - c) < 0.5 * (b - a)
                        # AND
                        # (s - c) < 0.5 * (a - b) -- wait, signs.
                        
                        # Better check from Brent's original:
                        # (s - a) < 0.5 * (b - a) is not sufficient.
                        # Let's check if s is in the "middle" part.
                        # If (s - a) < 0.5 * (b - a) and (b - s) < 0.5 * (b - a) -> s is in middle half.
                        # This is a bit strict.
                        
                        # Standard check:
                        # (s - c) < 0.5 * (b - a)  (if b > a)
                        # AND
                        # (s - c) < 0.5 * (a - b)  (if a > b)
                        
                        # Actually, simpler:
                        # if (s - a) < 0.5 * (b - a) and (s - b) < 0.5 * (a - b):
                        # This checks if s is within the central 50% of the interval?
                        
                        # Let's use the check:
                        # (s - c) < 0.5 * (b - a)
                        # AND
                        # (s - c) < 0.5 * (a - b)
                        # This ensures s is not too far from c relative to the bracket size.
                        
                        if (s - c) < 0.5 * (b - a) and (s - c) < 0.5 * (a - b):
                            use_iqi = True
                        else:
                            use_iqi = False
                    else:
                        use_iqi = False
                except (ZeroDivisionError, OverflowError, ValueError):
                    use_iqi = False

        if not use_iqi:
            # Try Secant
            # s = b - fb * (b - a) / (fb - fa)
            denom = fb - fa
            if abs(denom) > 1e-15:
                s_secant = b - fb * (b - a) / denom
                min_ab = min(a, b)
                max_ab = max(a, b)
                if min_ab < s_secant < max_ab:
                    s = s_secant
                    use_iqi = False # Mark as secant used, but logic for update is same
                else:
                    # Use Bisection
                    s = (a + b) / 2.0
                    use_iqi = False
            else:
                # Use Bisection
                s = (a + b) / 2.0
                use_iqi = False

        # Evaluate f(s)
        try:
            fs = f(s)
        except Exception:
            # If evaluation fails, try bisection point if s wasn't bisection
            if s != (a + b) / 2.0:
                s = (a + b) / 2.0
                try:
                    fs = f(s)
                except Exception:
                    break
            else:
                break
                
        if not math.isfinite(fs):
            break

        if abs(fs) <= 1e-8:
            return s

        # Update bracket
        if fa * fs <= 0:
            # Root in [a, s]
            b = s
            fb = fs
        else:
            # Root in [s, b]
            a = s
            fa = fs

        # Update best point c
        if abs(fs) < abs(fc):
            c = s
            fc = fs
        # Also check if a or b became better than c (handled at start of next loop, but let's do it here for efficiency)
        if abs(fa) < abs(fc):
            c = a
            fc = fa
        if abs(fb) < abs(fc):
            c = b
            fc = fb

    # Return best found
    if abs(fc) < abs(fb):
        return c
    else:
        return b

# EVOLVE-BLOCK-END
