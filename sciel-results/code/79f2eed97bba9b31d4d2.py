import time
import math

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using a multi-strategy approach:
    1. Quick check at x0
    2. Local secant/Newton for fast convergence
    3. Bracket expansion + Brent's method for robustness
    Deadline guard to avoid timeout.
    """
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin from 150s limit
    TOL = 1e-8
    TOL_STRICT = 1e-12
    
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
    
    # Strategy 1: Secant method (1 new eval per step after first 2)
    # Use secant with a small perturbation to get second point
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
    
    # Secant iterations
    x0_s = x
    x1_s = x1
    f0 = fx
    f1 = fx1
    
    best_x = x if abs(fx) < abs(fx1) else x1
    best_f = min(abs(fx), abs(fx1))
    
    for i in range(30):
        if check_deadline():
            break
        if abs(f1) < abs(f0):
            x0_s, x1_s = x1_s, x0_s
            f0, f1 = f1, f0
        
        if abs(f1) <= TOL:
            return x1_s
        
        if f0 == f1:
            break
        
        denom = f1 - f0
        if denom == 0:
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
    
    if abs(f1) <= TOL:
        return x1_s
    
    # Strategy 2: Bracket expansion + Brent's method
    # Try to find a sign change by expanding from x0
    a = x
    fa = fx
    b = x1
    fb = fx1
    
    # If we already have a bracket, use Brent
    if fa * fb < 0:
        return _brent(f, a, fa, b, fb, TOL, check_deadline)
    
    # Expand to find bracket
    h_expand = 1e-4
    max_expand = 1e8
    found_bracket = False
    
    for _ in range(60):
        if check_deadline():
            break
        h_expand *= 2
        if h_expand > max_expand:
            break
        
        # Try positive direction
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
        
        # Try negative direction
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
    
    # Strategy 3: Try a few random perturbations around x0
    import random
    random.seed(42)
    for _ in range(20):
        if check_deadline():
            break
        scale = random.uniform(1e-3, 1.0) * max(1.0, abs(x))
        xr = x + random.uniform(-scale, scale)
        try:
            fxr = f(xr)
        except Exception:
            continue
        if not math.isfinite(fxr):
            continue
        if abs(fxr) <= TOL:
            return xr
        if abs(fxr) < best_f:
            best_f = abs(fxr)
            best_x = xr
    
    # Return best found
    if abs(f(best_x)) <= TOL:
        return best_x
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
