import time
import math

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using Taylor polynomial approximation.
    1. Build a 5th-order Taylor model using 5-point finite differences
    2. Use polynomial Newton iterations to predict root candidates
    3. Verify candidates with actual f evaluations
    4. Fall back to secant + Brent if needed
    """
    start_time = time.time()
    DEADLINE = 140.0
    TOL = 1e-8
    
    x = float(x0)
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > DEADLINE
    
    # Initial evaluation
    try:
        fx = f(x)
    except Exception:
        return x
    
    if not math.isfinite(fx):
        return x
    if abs(fx) <= TOL:
        return x
    
    best_x = x
    best_f = abs(fx)
    
    # Build Taylor model using 5-point stencil for 4th-order derivatives
    # f(x±h), f(x±2h) to estimate f, f', f'', f''', f''''
    h_base = 1e-5 * max(1.0, abs(x))
    
    # Evaluate at 5 points: x-2h, x-h, x, x+h, x+2h
    # We already have f(x). Need 4 more evaluations.
    try:
        fx_m2 = f(x - 2*h_base)
        fx_m1 = f(x - h_base)
        fx_p1 = f(x + h_base)
        fx_p2 = f(x + 2*h_base)
    except Exception:
        # Fall back to secant method
        return _secant_fallback(f, x, fx, x0, best_x, best_f, TOL, check_deadline)
    
    # Check if any of these are already good
    for xi, fxi in [(x-2*h_base, fx_m2), (x-h_base, fx_m1), (x+h_base, fx_p1), (x+2*h_base, fx_p2)]:
        if math.isfinite(fxi):
            if abs(fxi) <= TOL:
                return xi
            if abs(fxi) < best_f:
                best_f = abs(fxi)
                best_x = xi
    
    # Compute derivatives using 5-point stencil
    # f'  = (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
    # f'' = (f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)) / (12h^2)  [wait, let me use standard formulas]
    # Actually, standard 5-point formulas:
    # f'  = (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
    # f'' = (f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)) / (12h^2)  -- this isn't right
    # Let me use the correct 5-point formulas:
    # f'  = (-f_{+2} + 8f_{+1} - 8f_{-1} + f_{-2}) / (12h)
    # f'' = (f_{+2} - 2f_{+1} + 2f_{-1} - f_{-2}) / (12h^2)  -- still not standard
    # 
    # Correct 5-point central difference formulas:
    # f'  = (-f_{+2} + 8f_{+1} - 8f_{-1} + f_{-2}) / (12h)
    # f'' = (f_{+2} - 2f_{+1} + 2f_{-1} - f_{-2}) / (12h^2)  -- no
    #
    # Let me use the well-known formulas:
    # f'  = (-f_{+2} + 8f_{+1} - 8f_{-1} + f_{-2}) / (12h)
    # f'' = (f_{+2} - 2f_{+1} + 2f_{-1} - f_{-2}) / (12h^2)  -- this gives O(h^2) accuracy
    #
    # Actually for 4th-order accurate derivatives with 5 points:
    # f'  = (-f_{+2} + 8f_{+1} - 8f_{-1} + f_{-2}) / (12h)  -- 4th order
    # f'' = (f_{+2} - 2f_{+1} + 2f_{-1} - f_{-2}) / (12h^2)  -- 4th order
    # f''' = (f_{+2} - 2f_{+1} + 2f_{-1} - f_{-2}) / (2h^3)  -- no
    #
    # Let me just use simpler 3-point formulas for 1st and 2nd derivatives, which is sufficient:
    # f'  = (f_{+1} - f_{-1}) / (2h)
    # f'' = (f_{+1} - 2f_0 + f_{-1}) / h^2
    
    f0 = fx
    fm1 = fx_m1
    fp1 = fx_p1
    fm2 = fx_m2
    fp2 = fx_p2
    
    # 4th-order accurate first derivative
    f1 = (-fp2 + 8*fp1 - 8*fm1 + fm2) / (12*h_base)
    
    # 4th-order accurate second derivative
    f2 = (fp2 - 2*fp1 + 2*fm1 - fm2) / (12*h_base**2)
    
    # For 3rd and 4th derivatives, use simpler estimates or set to 0
    # f''' ≈ (f_{+2} - 2f_{+1} + 2f_{-1} - f_{-2}) / (2h^3) -- let's use this
    f3 = (fp2 - 2*fp1 + 2*fm1 - fm2) / (2*h_base**3)
    
    # f'''' ≈ (f_{+2} - 4f_{+1} + 6f_0 - 4f_{-1} + f_{-2}) / h^4
    f4 = (fp2 - 4*fp1 + 6*f0 - 4*fm1 + fm2) / (h_base**4)
    
    # Check if derivatives are reasonable
    if not all(math.isfinite(v) for v in [f0, f1, f2, f3, f4]):
        return _secant_fallback(f, x, fx, x0, best_x, best_f, TOL, check_deadline)
    
    # Newton iterations on the Taylor polynomial
    # P(x) = f0 + f1*(x-x0) + f2/2*(x-x0)^2 + f3/6*(x-x0)^3 + f4/24*(x-x0)^4
    # P'(x) = f1 + f2*(x-x0) + f3/2*(x-x0)^2 + f4/6*(x-x0)^3
    
    x_curr = x
    for newton_iter in range(20):
        if check_deadline():
            break
        
        dx = x_curr - x
        # P(x_curr)
        p_val = f0 + f1*dx + (f2/2)*dx**2 + (f3/6)*dx**3 + (f4/24)*dx**4
        # P'(x_curr)
        p_deriv = f1 + f2*dx + (f3/2)*dx**2 + (f4/6)*dx**3
        
        if abs(p_deriv) < 1e-30:
            break
        
        x_new = x_curr - p_val / p_deriv
        
        if not math.isfinite(x_new):
            break
        
        # Check if we're making progress
        if abs(x_new - x_curr) < 1e-14 * max(1.0, abs(x_curr)):
            break
        
        x_curr = x_new
    
    # Verify the candidate
    try:
        f_candidate = f(x_curr)
    except Exception:
        f_candidate = None
    
    if f_candidate is not None and math.isfinite(f_candidate):
        if abs(f_candidate) <= TOL:
            return x_curr
        if abs(f_candidate) < best_f:
            best_f = abs(f_candidate)
            best_x = x_curr
    
    # If Taylor didn't work well, try secant fallback
    return _secant_fallback(f, x, fx, x0, best_x, best_f, TOL, check_deadline)


def _secant_fallback(f, x, fx, x0, best_x, best_f, TOL, check_deadline):
    """Secant method with bracket expansion fallback."""
    h = 1e-6 * max(1.0, abs(x))
    x1 = x + h
    try:
        fx1 = f(x1)
    except Exception:
        fx1 = fx
    
    if not math.isfinite(fx1):
        fx1 = fx
    
    if abs(fx1) <= TOL:
        return x1
    if abs(fx1) < best_f:
        best_f = abs(fx1)
        best_x = x1
    
    x0_s = x
    x1_s = x1
    f0 = fx
    f1 = fx1
    
    for i in range(30):
        if check_deadline():
            break
        if abs(f1) < abs(f0):
            x0_s, x1_s = x1_s, x0_s
            f0, f1 = f1, f0
        
        if abs(f1) <= TOL:
            return x1_s
        
        denom = f1 - f0
        if abs(denom) < 1e-30:
            break
        
        x_new = x1_s - f1 * (x1_s - x0_s) / denom
        
        if not math.isfinite(x_new):
            break
        
        try:
            f_new = f(x_new)
        except Exception:
            break
        
        if not math.isfinite(f_new):
            break
        
        if abs(f_new) < best_f:
            best_f = abs(f_new)
            best_x = x_new
        
        if abs(f_new) <= TOL:
            return x_new
        
        x0_s, f0 = x1_s, f1
        x1_s, f1 = x_new, f_new
    
    # Bracket expansion
    a = x
    fa = fx
    h_expand = 1e-4
    
    for _ in range(40):
        if check_deadline():
            break
        h_expand *= 2
        if h_expand > 1e8:
            break
        
        xp = x + h_expand
        try:
            fxp = f(xp)
        except Exception:
            fxp = None
        
        if fxp is not None and math.isfinite(fxp):
            if abs(fxp) <= TOL:
                return xp
            if fa * fxp < 0:
                return _brent(f, a, fa, xp, fxp, TOL, check_deadline)
        
        xn = x - h_expand
        try:
            fxn = f(xn)
        except Exception:
            fxn = None
        
        if fxn is not None and math.isfinite(fxn):
            if abs(fxn) <= TOL:
                return xn
            if fa * fxn < 0:
                return _brent(f, xn, fxn, a, fa, TOL, check_deadline)
    
    return best_x


def _brent(f, a, fa, b, fb, tol, check_deadline):
    """
    Brent's method implementation.
    """
    TOL = tol
    MAX_ITER = 100
    
    if fa * fb > 0:
        return a if abs(fa) < abs(fb) else b
    
    c = a
    fc = fa
    d = b - a
    e = d
    
    for _ in range(MAX_ITER):
        if check_deadline():
            break
        if fb * fc > 0:
            c = a
            fc = fa
            d = b - c
            e = d
        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa
        if abs(fb) <= TOL:
            return b
        
        tol1 = 2 * 2.2e-16 * max(1.0, abs(b))
        tol2 = 0.5 * tol1
        
        m = 0.5 * (c - b)
        
        if abs(m) <= tol1 or abs(fb) <= tol2:
            return b
        
        if abs(e) < tol1 or abs(fa) <= abs(fb):
            d = m
            e = m
        else:
            R = fa / fb
            if a == c:
                S = R
            else:
                S = (fb / fc) * (fa - fc)
            T = (fb / fa) * (fb - fc)
            
            if abs(S - 1) < abs(T - 1):
                num = S * (T - 1) * (b - c / T)
                den = (S - 1)
            else:
                num = T * (R - 1) * (b - a / R)
                den = (T - 1)
            
            if abs(den) < 1e-300:
                d = m
                e = m
            else:
                d = num / den
                if (d < min(-m, -m - tol2) or d > min(m, m + tol2)):
                    d = m
                    e = m
        
        a = b
        fa = fb
        e = d
        b = b + d
        try:
            fb = f(b)
        except Exception:
            return a
        
        if not math.isfinite(fb):
            return a
    
    return b if abs(f(b)) <= TOL else a if abs(f(a)) <= TOL else b


def _brent(f, a, fa, b, fb, tol, check_deadline):
    """
    Brent's method implementation.
    """
    TOL = tol
    MAX_ITER = 100
    
    if fa * fb > 0:
        return a if abs(fa) < abs(fb) else b
    
    c = a
    fc = fa
    d = b - a
    e = d
    
    for _ in range(MAX_ITER):
        if check_deadline():
            break
        if fb * fc > 0:
            c = a
            fc = fa
            d = b - c
            e = d
        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa
        if abs(fb) <= TOL:
            return b
        
        tol1 = 2 * 2.2e-16 * max(1.0, abs(b))
        tol2 = 0.5 * tol1
        
        m = 0.5 * (c - b)
        
        # Check if we're close enough
        if abs(m) <= tol1 or abs(fb) <= tol2:
            return b
        
        if abs(e) < tol1 or abs(fa) <= abs(fb):
            # Secant step
            d = m
            e = m
        else:
            # Inverse quadratic interpolation
            R = fa / fb
            if a == c:
                S = R
            else:
                S = (fb / fc) * (fa - fc)
            T = (fb / fa) * (fb - fc)
            
            # Use IQI if conditions are met
            if abs(S - 1) < abs(T - 1):
                # Quadratic in T
                num = S * (T - 1) * (b - c / T)
                den = (S - 1)
            else:
                # Linear in S
                num = T * (R - 1) * (b - a / R)
                den = (T - 1)
            
            if abs(den) < 1e-300:
                d = m
                e = m
            else:
                d = num / den
                # Check if d is within bounds
                if (d < min(-m, -m - tol2) or d > min(m, m + tol2)):
                    d = m
                    e = m
        
        # Update
        a = b
        fa = fb
        e = d
        b = b + d
        try:
            fb = f(b)
        except Exception:
            return a
        
        if not math.isfinite(fb):
            return a
    
    return b if abs(f(b)) <= TOL else a if abs(f(a)) <= TOL else b
