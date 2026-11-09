def find_root(f, x0):
    MAX_CALLS = 480
    calls = [0]
    
    def feval(x):
        if calls[0] >= MAX_CALLS:
            return None
        calls[0] += 1
        return f(x)
    
    x = float(x0)
    fx = feval(x)
    if fx is None:
        return x
    if abs(fx) <= 1e-8:
        return x
    
    # Phase 1: Bracket
    step = 1e-2 * max(1.0, abs(x))
    a, b = x - step, x + step
    fa = feval(a)
    fb = feval(b)
    
    if fa is None or fb is None:
        return x
        
    if abs(fa) <= 1e-8: return a
    if abs(fb) <= 1e-8: return b
    
    if fa * fb > 0:
        expanded = 0
        while fa * fb > 0 and expanded < 50 and calls[0] < MAX_CALLS - 10:
            step *= 2.0
            a_new, b_new = x - step, x + step
            fa_new = feval(a_new)
            fb_new = feval(b_new)
            if fa_new is None or fb_new is None:
                break
            if abs(fa_new) <= 1e-8: return a_new
            if abs(fb_new) <= 1e-8: return b_new
            if fa_new * fb_new < 0:
                a, b, fa, fb = a_new, b_new, fa_new, fb_new
                break
            a, b = a_new, b_new
            fa, fb = fa_new, fb_new
            expanded += 1
            
        if fa * fb > 0:
            # Fallback: Secant method
            # This case is rare for "fixed smooth polynomials" with good x0
            # But we need to handle it.
            # Use Secant from x and x+step
            c = x + step
            fc = feval(c)
            if fc is None: return x
            for _ in range(100):
                if calls[0] >= MAX_CALLS: break
                if fa == fb: break
                # Secant step from a, b
                # Actually we have a, b, fa, fb (same sign)
                # Let's use x and c
                # This is getting messy. 
                # Let's just return best guess if no bracket.
                # But wait, if we expand enough, we MUST find a bracket for a polynomial 
                # unless the root is complex or we missed it. 
                # Given "fixed smooth polynomial", real roots are likely.
                pass

    # Ensure |fb| < |fa|
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
        
    c = a
    d = a
    e = 0.0
    fc = fa # f(c) where c is the old a
    
    for _ in range(200):
        if calls[0] >= MAX_CALLS:
            break
            
        if abs(fb) <= 1e-8:
            return b
        if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
            return b
            
        # 1. If f(a)*f(c) < 0, swap
        # We need f(c). c is the previous a.
        # In standard Brent:
        # If f(a)*f(c) < 0: b=a, a=c, c=b, d=b, e=b-a
        # Here c is initialized to a. So f(c) is fa.
        # So fa*fc is fa*fa which is positive. No swap initially.
        
        # Actually, let's just use the standard variables:
        # a, b: bracket
        # c: previous a
        # d: previous b
        # e: last step
        
        # Check condition for swapping a and c
        # f(a) * f(c) < 0 ?
        # We don't have f(c) stored separately from fa? 
        # c is the point that was 'a' in the previous step.
        # Let's maintain fc explicitly.
        
        # Initial: a, b, fa, fb. c=a, fc=fa.
        
        if fa * fc < 0:
            b, a = a, b
            fb, fa = fa, fb
            c, d = b, a
            e = b - a
            # Update fc? c is now the new b? No.
            # Standard:
            # if f(a)*f(c) < 0:
            #   b = a; a = c; c = b
            #   d = b; e = b - a
            # So c becomes the old b? No, c becomes the new b?
            # Let's trace:
            # a, b are bracket. c is previous a.
            # If f(a)*f(c) < 0, then c and a have different signs? No, f(a) and f(c) have different signs.
            # This means c is on the other side of the root from a?
            # Actually, this step ensures that b is the point with the smaller |f|.
            
            # Let's stick to a simpler, robust implementation often found in textbooks.
            # 
            # 1. If |f(a)| < |f(b)|: swap a, b
            # 2. Set c = a, d = a
            # 3. Loop:
            #    a, b are bracket.
            #    If |f(b)| < |f(a)|:
            #       # b is better
            #       pass
            #    else:
            #       a, b = b, a
            #       fa, fb = fb, fa
            #    
            #    # Now |fb| <= |fa|
            #    
            #    # Calculate next point p
            #    # Use IQI or Secant or Bisection
            #    
            #    # Check if p is valid
            #    # If valid, move to p
            #    # Else, bisection
            
            # This "Swap to keep |fb| < |fa|" approach is simpler to implement correctly.
            # Let's use this.
            
            break # Exit the "if fa*fc<0" block logic and use the simpler swap logic in the main loop
            
    # Let's restart the loop with the simpler logic
    
    c = a
    d = a
    e = 0.0
    
    for _ in range(200):
        if calls[0] >= MAX_CALLS:
            break
            
        if abs(fb) <= 1e-8:
            return b
        if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
            return b
            
        # Ensure |fb| < |fa|
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
            
        # Save previous values
        c = a
        d = a
        e_prev = e
        
        # Calculate next step
        # 1. Try Inverse Quadratic Interpolation (IQI)
        # Requires 3 points: a, b, c (where c is previous a)
        # But we only have a, b current. We need history.
        # Let's maintain:
        # a, b: current bracket
        # c: previous a
        # d: previous b
        # e: step size in previous iteration (d - c? or b - a?)
        
        # Let's use the variables:
        # a, b: bracket
        # c: previous a
        # d: previous b
        # e: last step (b - a) from previous iteration?
        
        # Standard Brent variables:
        # a, b: bracket
        # c: previous a
        # d: previous b
        # e: last step
        
        # 1. If f(a)*f(c) < 0:
        #    b = a; a = c; c = b; d = b; e = b - a
        # 2. If |c| < |b|:
        #    b = a; a = c; c = b; d = b; e = b - a
        # 3. m = e; e = d; d = b
        # 4. If |e| < tol or |f(b)| < tol: return b
        # 5. If |f(a)| < |f(b)|: S = f(b)/f(a) else S = 1
        # 6. If |e| >= tol and |S - 1| > tol:
        #    # Try IQI
        #    # If IQI fails, try Secant
        #    # If both fail, Bisection
        # 7. Else Bisection
        
        # This is complex to get right without a reference.
        # Given the constraints, a simpler **Illinois Method** (which is already there) 
        # or **Bisection** is safer?
        # The current code uses Illinois.
        # Maybe the improvement is in the **Bracketing Phase** or **Fallback**?
        
        # Let's look at the score: 0.8565.
        # The penalty is for calls.
        # Illinois is linear convergence.
        # Brent's is superlinear.
        # If I can implement Brent's correctly, it should use fewer calls.
        
        # Let's try a different approach: **Newton-Raphson with Bisection Fallback**.
        # Since f is a polynomial, f' is a polynomial.
        # We can approximate f' with central difference.
        # h = 1e-5 * max(1, abs(x))
        # f'(x) ~ (f(x+h) - f(x-h)) / (2h)
        # This costs 2 extra calls per Newton step.
        # Newton step: x_new = x - f(x)/f'(x)
        # If x_new is outside bracket [a,b], use Bisection.
        # This is very robust and fast (quadratic convergence).
        
        # Let's implement this.
        
        break # Break out of the fake loop above and implement Newton+Bisection
        
    # --- New Approach: Damped Newton with Bisection Fallback ---
    
    # We have bracket [a, b] with fa*fb < 0
    # Ensure fa < 0 < fb or vice versa.
    # Let's define:
    # low, high: bracket
    # f_low, f_high
    
    if fa * fb > 0:
        # No bracket. Use Secant.
        # This is a fallback.
        x1 = x
        x2 = x + 1e-4 * max(1.0, abs(x))
        f1 = fx
        f2 = feval(x2)
        if f2 is None: return x
        for _ in range(100):
            if calls[0] >= MAX_CALLS: break
            if f1 == f2: break
            x3 = x2 - f2 * (x2 - x1) / (f2 - f1)
            f3 = feval(x3)
            if f3 is None: break
            if abs(f3) <= 1e-8: return x3
            x1, f1 = x2, f2
            x2, f2 = x3, f3
        return x2
        
    # We have bracket
    # Let's call the endpoints lo, hi
    if fa < 0:
        lo, hi = a, b
        flo, fhi = fa, fb
    else:
        lo, hi = b, a
        flo, fhi = fb, fa
        
    # Newton's method
    x = (lo + hi) / 2.0 # Start in middle
    fx = feval(x)
    if fx is None: return x
    if abs(fx) <= 1e-8: return x
    
    # If fx has same sign as flo, x is in [x, hi]
    # If fx has same sign as fhi, x is in [lo, x]
    # Update bracket
    if flo * fx < 0:
        hi = x
        fhi = fx
    else:
        lo = x
        flo = fx
        
    for _ in range(100):
        if calls[0] >= MAX_CALLS:
            break
            
        # Calculate derivative
        h = 1e-6 * max(1.0, abs(x))
        xp = x + h
        xm = x - h
        f_p = feval(xp)
        f_m = feval(xm)
        if f_p is None or f_m is None:
            break
            
        if abs(f_p - f_m) < 1e-300:
            break
            
        df = (f_p - f_m) / (2.0 * h)
        
        if abs(df) < 1e-300:
            break
            
        x_new = x - fx / df
        
        # Check if x_new is within bracket [lo, hi]
        if x_new <= lo or x_new >= hi:
            # Use Bisection
            x_new = (lo + hi) / 2.0
            f_new = feval(x_new)
            if f_new is None:
                break
            if abs(f_new) <= 1e-8:
                return x_new
            if flo * f_new < 0:
                hi = x_new
                fhi = f_new
            else:
                lo = x_new
                flo = f_new
        else:
            f_new = feval(x_new)
            if f_new is None:
                break
            if abs(f_new) <= 1e-8:
                return x_new
            if flo * f_new < 0:
                hi = x_new
                fhi = f_new
            else:
                lo = x_new
                flo = f_new
                
        x = x_new
        fx = f_new
        
        if abs(hi - lo) < 1e-15 * max(1.0, abs(lo), abs(hi)):
            break
            
    return x
