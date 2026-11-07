"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

DEADLINE = time.time() + 110.0  # Safety margin before 120s limit

def _check_deadline():
    return time.time() < DEADLINE

def find_root(f, x0):
    x = float(x0)
    best_x = x
    best_fx = None
    
    # Phase 1: Fast convergence using Newton with central difference
    for _ in range(40):
        if not _check_deadline():
            break
        if not math.isfinite(x):
            break
        try:
            fx = f(x)
        except:
            break
        if not math.isfinite(fx):
            break
        
        # Track best
        if best_fx is None or abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx
        
        if abs(fx) <= 1e-10:  # Tighter internal tolerance for polish
            # Phase 2: High-precision polish
            x = _polish(f, x, fx)
            return x
        
        # Central difference for derivative
        h = 1e-5 * max(1.0, abs(x))
        try:
            f_xp = f(x + h)
            f_xm = f(x - h)
        except:
            break
        if not math.isfinite(f_xp) or not math.isfinite(f_xm):
            break
            
        d = (f_xp - f_xm) / (2.0 * h)
        
        if abs(d) < 1e-14:
            # Derivative too small, try secant-like step
            step = -fx * 1e-3
            x_new = x + step
        else:
            x_new = x - fx / d
        
        if not math.isfinite(x_new):
            break
            
        # Damping: prevent large jumps
        step = x_new - x
        if abs(step) > 2.0:
            x_new = x + (2.0 if step > 0 else -2.0)
            
        x = x_new
    
    # If we didn't converge tightly, try secant method as fallback
    if best_fx is not None and abs(best_fx) > 1e-8 and _check_deadline():
        x = _secant_fallback(f, x, best_x, best_fx)
        # Update best after secant
        try:
            fx_check = f(x)
            if math.isfinite(fx_check) and (best_fx is None or abs(fx_check) < abs(best_fx)):
                best_x = x
                best_fx = fx_check
        except:
            pass
    
    # Final polish if close
    if best_fx is not None and abs(best_fx) <= 1e-6 and _check_deadline():
        x = _polish(f, x, best_fx)
        try:
            fx_check = f(x)
            if math.isfinite(fx_check) and (best_fx is None or abs(fx_check) < abs(best_fx)):
                best_x = x
                best_fx = fx_check
        except:
            pass
    
    # If still not converged, try bracketing + bisection/interpolation as last resort
    if best_fx is not None and abs(best_fx) > 1e-8 and _check_deadline():
        x_bracketed = _bracket_and_solve(f, best_x, best_fx)
        if x_bracketed is not None:
            try:
                fx_br = f(x_bracketed)
                if math.isfinite(fx_br) and (best_fx is None or abs(fx_br) < abs(best_fx)):
                    best_x = x_bracketed
                    best_fx = fx_br
            except:
                pass
    
    return best_x

def _polish(f, x, fx):
    """High-precision Newton polish with tighter tolerance."""
    for _ in range(20):
        if not _check_deadline():
            break
        if not math.isfinite(x) or fx is None or not math.isfinite(fx):
            break
        if abs(fx) <= 1e-10:
            break
        
        # Use smaller h for higher precision
        h = 1e-7 * max(1.0, abs(x))
        try:
            f_xp = f(x + h)
            f_xm = f(x - h)
        except:
            break
        if not math.isfinite(f_xp) or not math.isfinite(f_xm):
            break
        
        d = (f_xp - f_xm) / (2.0 * h)
        if abs(d) < 1e-16:
            break
        
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        
        # Small damping for polish
        step = x_new - x
        if abs(step) > 0.1:
            x_new = x + (0.1 if step > 0 else -0.1)
        
        x = x_new
        try:
            fx = f(x)
        except:
            break
        if not math.isfinite(fx):
            break
    
    return x

def _secant_fallback(f, x, x_prev, fx_prev):
    """Secant method fallback when Newton struggles."""
    for _ in range(30):
        if not _check_deadline():
            break
        if not math.isfinite(x) or not math.isfinite(x_prev):
            break
        try:
            fx = f(x)
        except:
            break
        if not math.isfinite(fx):
            break
        if abs(fx) <= 1e-10:
            return x
        
        if abs(fx) < abs(fx_prev):
            x_prev, fx_prev = x, fx
        elif abs(fx_prev) < 1e-10:
            return x_prev
        
        denom = fx - fx_prev
        if abs(denom) < 1e-15:
            break
        
        x_new = x - fx * (x - x_prev) / denom
        if not math.isfinite(x_new):
            break
        
        x_prev, fx_prev = x, fx
        x = x_new
    
    return x

def _bracket_and_solve(f, x_start, fx_start):
    """Try to find a bracket and solve using bisection/interpolation."""
    if not _check_deadline():
        return None
    if fx_start is None or not math.isfinite(fx_start):
        return None
    
    a = x_start
    fa = fx_start
    step = 1.0
    b = None
    fb = None
    
    # Expand to find a sign change
    for _ in range(30):
        if not _check_deadline():
            break
        b = a + step
        try:
            fb = f(b)
        except:
            step *= 2.0
            continue
        if not math.isfinite(fb):
            step *= 2.0
            continue
        if fa * fb <= 0:
            break
        # If no sign change, try the other direction
        b = a - step
        try:
            fb = f(b)
        except:
            step *= 2.0
            continue
        if not math.isfinite(fb):
            step *= 2.0
            continue
        if fa * fb <= 0:
            break
        step *= 2.0
    else:
        return None
    
    # Now we have a bracket [a, b] with fa * fb <= 0
    # Ensure a is the left endpoint
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
    
    best_x = a if abs(fa) < abs(fb) else b
    best_f = min(abs(fa), abs(fb))
    
    # Safeguarded inverse interpolation
    for _ in range(50):
        if not _check_deadline():
            break
        if abs(fb) <= 1e-8:
            return b
        if abs(fa) <= 1e-8:
            return a
        
        denom = fb - fa
        if abs(denom) < 1e-300:
            x_new = (a + b) / 2.0
        else:
            x_new = (a * fb - b * fa) / denom
        
        # Safeguard: x_new must be in (a, b)
        if not (a < x_new < b):
            x_new = (a + b) / 2.0
        
        try:
            f_new = f(x_new)
        except:
            x_new = (a + b) / 2.0
            try:
                f_new = f(x_new)
            except:
                break
        if not math.isfinite(f_new):
            x_new = (a + b) / 2.0
            try:
                f_new = f(x_new)
            except:
                break
            if not math.isfinite(f_new):
                break
        
        if abs(f_new) < best_f:
            best_x = x_new
            best_f = abs(f_new)
        
        if abs(f_new) <= 1e-8:
            return x_new
        
        # Update bracket
        if fa * f_new <= 0:
            b = x_new
            fb = f_new
        else:
            a = x_new
            fa = f_new
        
        # Check convergence
        if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
            break
    
    if abs(fb) <= abs(fa):
        return b
    else:
        return a
# EVOLVE-BLOCK-END
