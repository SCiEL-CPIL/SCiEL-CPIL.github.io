"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Find a root x such that |f(x)| <= 1e-8 using at most 500 evaluations of f.
    
    Strategy:
    1. Evaluate f(x0). If |f(x0)| <= 1e-8, return x0.
    2. Try a few secant/Newton steps with finite differences to get close.
    3. Establish a bracket [a, b] where f(a) and f(b) have opposite signs
       using adaptive step-doubling from the best point found.
    4. Apply the Illinois method:
       - Compute secant point c.
       - If c is on the same side as the last replaced endpoint, shrink
         that side by a factor of 0.5 instead of using c.
       - Track best x and return when |f(x)| <= 1e-8.
    """
    max_evals = 500
    evals_used = [0]
    
    def feval(x):
        if evals_used[0] >= max_evals:
            raise RuntimeError("Too many evaluations")
        val = f(x)
        evals_used[0] += 1
        return val
    
    x = float(x0)
    
    # Phase 1: Evaluate at x0
    best_x = x
    best_fx = feval(x)
    best_abs = abs(best_fx)
    
    if best_abs <= 1e-8:
        return x
    
    # Phase 2: Try secant/Newton steps with finite differences
    x_prev = None
    fx_prev = None
    for i in range(15):
        if evals_used[0] >= max_evals - 10:
            break
        h = 1e-7 * max(1.0, abs(x))
        try:
            fx_h = feval(x + h)
        except Exception:
            break
        if not math.isfinite(fx_h):
            break
        d = (fx_h - best_fx) / h
        if abs(d) < 1e-15:
            break
        x_new = x - best_fx / d
        if not math.isfinite(x_new):
            break
        if abs(x_new) > 1e10 * max(1.0, abs(x)):
            break
        try:
            fx_new = feval(x_new)
        except Exception:
            break
        if not math.isfinite(fx_new):
            break
        abs_fx_new = abs(fx_new)
        if abs_fx_new < best_abs:
            best_x = x_new
            best_abs = abs_fx_new
            if best_abs <= 1e-8:
                return x_new
        # Use secant for next step
        x_prev, fx_prev = x, best_fx
        x, best_fx = x_new, fx_new
        best_abs = abs(best_fx)
        if best_abs <= 1e-8:
            return x
    
    if best_abs <= 1e-8:
        return best_x
    
    # Phase 3: Establish a bracket using step-doubling
    a = best_x
    fa = best_fx
    
    b = None
    fb = None
    
    # Try step-doubling in both directions
    step = 1e-4
    for direction in [1.0, -1.0]:
        for i in range(60):
            if evals_used[0] >= max_evals - 5:
                break
            b = a + direction * step
            try:
                fb = feval(b)
            except Exception:
                break
            if not math.isfinite(fb):
                step *= 2.0
                continue
            if fa * fb < 0:
                break
            step *= 2.0
        if b is not None and fb is not None and math.isfinite(fb) and fa * fb < 0:
            break
    
    # If we couldn't find a bracket, return the best we have
    if b is None or fb is None or not math.isfinite(fb) or fa * fb >= 0:
        return best_x
    
    # Phase 4: Illinois method
    a_curr = a
    b_curr = b
    fa_curr = fa
    fb_curr = fb
    
    # Update best
    if abs(fa_curr) < best_abs:
        best_x = a_curr
        best_abs = abs(fa_curr)
    if abs(fb_curr) < best_abs:
        best_x = b_curr
        best_abs = abs(fb_curr)
    
    if best_abs <= 1e-8:
        return best_x
    
    # Illinois method:
    # last_side: which endpoint was replaced in the previous iteration ('a' or 'b')
    # If the secant point c falls on the same side as last_side, shrink that side by 0.5
    last_side = None
    shrink_factor = 0.5
    
    for i in range(450):
        if evals_used[0] >= max_evals - 1:
            break
        
        # Compute secant point
        denom = fb_curr - fa_curr
        if abs(denom) < 1e-15:
            c = (a_curr + b_curr) / 2.0
        else:
            c = a_curr - fa_curr * (b_curr - a_curr) / denom
        
        # Check if c is on the same side as last_side
        # c is on 'a' side if it's closer to a_curr than to b_curr, OR
        # more precisely: if f(c) would have same sign as f(a_curr)
        # But we haven't evaluated f(c) yet, so we use position:
        # c is on 'a' side if (c - a_curr) / (b_curr - a_curr) < 0.5
        # Actually, the standard Illinois check is based on which endpoint
        # gets replaced. Let's use: if c is closer to a_curr, it's on 'a' side.
        
        if b_curr != a_curr:
            frac = (c - a_curr) / (b_curr - a_curr)
        else:
            frac = 0.5
        
        # If last_side is 'a' and c is on the 'a' side (frac < 0.5), shrink a-side
        # If last_side is 'b' and c is on the 'b' side (frac > 0.5), shrink b-side
        if last_side == 'a' and frac < 0.5:
            # Shrink a-side: new a = a + shrink_factor * (b - a)
            c = a_curr + shrink_factor * (b_curr - a_curr)
        elif last_side == 'b' and frac > 0.5:
            # Shrink b-side: new b = b - shrink_factor * (b - a)
            c = b_curr - shrink_factor * (b_curr - a_curr)
        
        # Clamp c to strictly inside (a_curr, b_curr)
        if c <= a_curr:
            c = a_curr + 1e-15 * (b_curr - a_curr) if b_curr > a_curr else a_curr + 1e-15
        elif c >= b_curr:
            c = b_curr - 1e-15 * (b_curr - a_curr) if b_curr > a_curr else b_curr - 1e-15
        
        # Evaluate f(c)
        try:
            fc = feval(c)
        except Exception:
            break
        
        if not math.isfinite(fc):
            break
        
        abs_fc = abs(fc)
        if abs_fc < best_abs:
            best_x = c
            best_abs = abs_fc
            if best_abs <= 1e-8:
                return c
        
        # Update bracket
        if fa_curr * fc <= 0:
            # Root in [a_curr, c], replace b
            b_curr = c
            fb_curr = fc
            last_side = 'b'
        else:
            # Root in [c, b_curr], replace a
            a_curr = c
            fa_curr = fc
            last_side = 'a'
        
        # Check convergence
        if abs(b_curr - a_curr) < 1e-14 * max(1.0, abs(a_curr), abs(b_curr)):
            break
    
    return best_x

# EVOLVE-BLOCK-END
