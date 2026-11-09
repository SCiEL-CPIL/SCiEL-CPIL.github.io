def find_root(f, x0):
    MAX_CALLS = 480
    calls = [0]
    
    def feval(x):
        if calls[0] >= MAX_CALLS:
            return None
        calls[0] += 1
        try:
            return f(x)
        except Exception:
            return None

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
            # No bracket found. Fallback to Secant
            c = x + step
            fc = feval(c)
            if fc is None: return x
            if fc == fa: return x
            for _ in range(50):
                if calls[0] >= MAX_CALLS: break
                denom = fc - fa
                if denom == 0: break
                x_new = c - fc * (c - x) / denom
                f_new = feval(x_new)
                if f_new is None: break
                if abs(f_new) <= 1e-8: return x_new
                x, fa = c, fc
                c, fc = x_new, f_new
            return c

    # Phase 2: Illinois Method
    # Ensure |fb| < |fa| for stability in some implementations, though Illinois is symmetric
    # We just need fa * fb < 0
    
    for _ in range(100):
        if calls[0] >= MAX_CALLS:
            break
            
        if abs(fb) <= 1e-8:
            return b
        if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
            return b

        # Secant step
        denom = fb - fa
        if denom == 0:
            c = (a + b) / 2.0
            fc = feval(c)
            if fc is None: break
            if abs(fc) <= 1e-8: return c
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
                fa *= 0.5
            continue

        c = b - fb * (b - a) / denom
        fc = feval(c)
        if fc is None:
            break
            
        if abs(fc) <= 1e-8:
            return c

        if fa * fc < 0:
            # f(a) and f(c) have opposite signs.
            # f(b) and f(c) have same signs.
            # Update b = c, halve f(a)
            b = c
            fb = fc
            fa = fa / 2.0
        else:
            # f(a) and f(c) have same signs.
            # f(b) and f(c) have opposite signs.
            # Update a = c, halve f(b)
            a = c
            fa = fc
            fb = fb / 2.0

    return b
