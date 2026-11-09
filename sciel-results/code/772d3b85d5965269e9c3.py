"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
    
    best_x = x
    best_fx = fx
    
    # Phase 1: Find a bracket around x0
    step = 1e-2 * max(1.0, abs(x))
    a, b = x - step, x + step
    fa = feval(a)
    fb = feval(b)
    if fa is not None and abs(fa) < abs(best_fx):
        best_x, best_fx = a, fa
    if fb is not None and abs(fb) < abs(best_fx):
        best_x, best_fx = b, fb
    
    expanded = 0
    while (fa is None or fb is None or fa * fb > 0) and expanded < 40 and calls[0] < MAX_CALLS - 5:
        step *= 2.0
        a_new, b_new = x - step, x + step
        fa_new = feval(a_new)
        fb_new = feval(b_new)
        if fa_new is not None and abs(fa_new) < abs(best_fx):
            best_x, best_fx = a_new, fa_new
        if fb_new is not None and abs(fb_new) < abs(best_fx):
            best_x, best_fx = b_new, fb_new
        if fa_new is not None:
            fa = fa_new
        if fb_new is not None:
            fb = fb_new
        a, b = a_new, b_new
        expanded += 1
    
    # Phase 2: If bracketed, use Illinois method
    if fa is not None and fb is not None and fa * fb <= 0:
        for _ in range(400):
            if calls[0] >= MAX_CALLS:
                break
            if abs(fa) < abs(fb):
                a, b, fa, fb = b, a, fb, fa
            denom = fb - fa
            if abs(denom) < 1e-300:
                c = (a + b) / 2.0
            else:
                c = b - fb * (b - a) / denom
            if c <= a or c >= b or abs(c - a) > 0.9 * abs(b - a) or abs(c - b) > 0.9 * abs(b - a):
                c = (a + b) / 2.0
            fc = feval(c)
            if fc is None:
                break
            if abs(fc) < abs(best_fx):
                best_x, best_fx = c, fc
            if abs(fc) <= 1e-8:
                return c
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
            if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
                break
        return best_x
    else:
        # Phase 2b: No bracket, use secant method
        a, b = x, x + 1e-4 * max(1.0, abs(x))
        fa, fb = fx, feval(b)
        if fb is None:
            return best_x
        if abs(fb) < abs(best_fx):
            best_x, best_fx = b, fb
        for _ in range(300):
            if calls[0] >= MAX_CALLS:
                break
            if fa == fb:
                break
            c = b - fb * (b - a) / (fb - fa)
            if abs(c - b) > 10 * max(abs(b - a), 1e-10):
                c = b + (b - a)
            fc = feval(c)
            if fc is None:
                break
            if abs(fc) < abs(best_fx):
                best_x, best_fx = c, fc
            if abs(fc) <= 1e-8:
                return c
            a, fa = b, fb
            b, fb = c, fc
            if fa * fb < 0:
                # Now bracketed, switch to Illinois
                for _ in range(300):
                    if calls[0] >= MAX_CALLS:
                        break
                    if abs(fa) < abs(fb):
                        a, b, fa, fb = b, a, fb, fa
                    denom = fb - fa
                    if abs(denom) < 1e-300:
                        c = (a + b) / 2.0
                    else:
                        c = b - fb * (b - a) / denom
                    if c <= a or c >= b or abs(c - a) > 0.9 * abs(b - a) or abs(c - b) > 0.9 * abs(b - a):
                        c = (a + b) / 2.0
                    fc = feval(c)
                    if fc is None:
                        break
                    if abs(fc) < abs(best_fx):
                        best_x, best_fx = c, fc
                    if abs(fc) <= 1e-8:
                        return c
                    if fa * fc < 0:
                        b, fb = c, fc
                    else:
                        a, fa = c, fc
                    if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
                        break
                return best_x
        return best_x
# EVOLVE-BLOCK-END
