"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START

def _safe_f(f, x):
    """Evaluate f(x) safely, returning inf on any error or non-finite result."""
    try:
        v = f(float(x))
        v = float(v)
        if not math.isfinite(v):
            return float('inf')
        return v
    except Exception:
        return float('inf')


def _newton_step(f, x, fx, h_base):
    """
    Perform one damped Newton step.
    Returns (x_new, fx_new, converged, failed).
    Uses central finite difference for derivative.
    """
    # Use central difference for better accuracy
    h = h_base * max(1.0, abs(x))
    # Adaptive h: shrink if x is very small
    if abs(x) < 1e-6:
        h = 1e-7
    
    fx_plus = _safe_f(f, x + h)
    fx_minus = _safe_f(f, x - h)
    
    if math.isinf(fx_plus) or math.isinf(fx_minus):
        # Try forward difference
        fx_plus = _safe_f(f, x + h)
        if math.isinf(fx_plus):
            return x, fx, False, True
        d = (fx_plus - fx) / h
    else:
        d = (fx_plus - fx_minus) / (2 * h)
    
    if abs(d) < 1e-14:
        return x, fx, False, True
    
    step = fx / d
    x_new = x - step
    
    # Damping: limit step size
    max_step = 5.0 * max(1.0, abs(x))
    if abs(step) > max_step:
        step = math.copysign(max_step, step)
        x_new = x - step
    
    fx_new = _safe_f(f, x_new)
    
    if math.isinf(fx_new):
        # Try a smaller step
        step2 = step * 0.1
        x_new2 = x - step2
        fx_new2 = _safe_f(f, x_new2)
        if math.isfinite(fx_new2) and abs(fx_new2) < abs(fx):
            x_new, fx_new = x_new2, fx_new2
        else:
            return x, fx, False, True
    
    if abs(fx_new) <= 1e-8:
        return x_new, fx_new, True, False
    
    return x_new, fx_new, False, False


def find_root(f, x0):
    start_time = time.monotonic()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    x = float(x0)
    fx = _safe_f(f, x)
    
    if math.isinf(fx):
        # Starting point is bad, try some offsets
        for offset in [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 10.0, -10.0]:
            x_try = x + offset
            fx_try = _safe_f(f, x_try)
            if math.isfinite(fx_try):
                if abs(fx_try) <= 1e-8:
                    return x_try
                x, fx = x_try, fx_try
                break
        else:
            return x  # All bad, return original
    
    if abs(fx) <= 1e-8:
        return x
    
    h_base = 1e-6
    
    # Main Newton loop with convergence detection
    prev_x = None
    prev_fx = None
    stall_count = 0
    best_x = x
    best_fx = abs(fx)
    
    for iteration in range(80):
        if time.monotonic() > deadline:
            break
        
        # Check for convergence
        if abs(fx) <= 1e-10:
            return x
        
        # Track best
        if abs(fx) < best_fx:
            best_fx = abs(fx)
            best_x = x
        
        # Detect stall
        if prev_x is not None:
            if abs(x - prev_x) < 1e-12 * max(1.0, abs(x)):
                stall_count += 1
            else:
                stall_count = 0
        
        prev_x = x
        prev_fx = fx
        
        # Try Newton step
        x_new, fx_new, converged, failed = _newton_step(f, x, fx, h_base)
        
        if converged:
            return x_new
        
        if not failed:
            # Accept if it improves
            if abs(fx_new) < abs(fx) * 1.5:
                x, fx = x_new, fx_new
            else:
                # Try a bisection-like step
                step = math.copysign(1e-3 * max(1.0, abs(x)), -fx)
                x_bis = x + step
                fx_bis = _safe_f(f, x_bis)
                if math.isfinite(fx_bis) and abs(fx_bis) < abs(fx):
                    x, fx = x_bis, fx_bis
                else:
                    # Accept the Newton step anyway to make progress
                    x, fx = x_new, fx_new
        else:
            # Newton failed, try bisection-like
            step = math.copysign(1e-3 * max(1.0, abs(x)), -fx)
            x_bis = x + step
            fx_bis = _safe_f(f, x_bis)
            if math.isfinite(fx_bis) and abs(fx_bis) < abs(fx):
                x, fx = x_bis, fx_bis
            else:
                # Try opposite direction
                step2 = math.copysign(1e-3 * max(1.0, abs(x)), fx)
                x_bis2 = x + step2
                fx_bis2 = _safe_f(f, x_bis2)
                if math.isfinite(fx_bis2) and abs(fx_bis2) < abs(fx):
                    x, fx = x_bis2, fx_bis2
                else:
                    # Just move a tiny bit
                    x += 1e-4 * math.copysign(1.0, fx)
                    fx = _safe_f(f, x)
                    if math.isinf(fx):
                        x -= 2e-4 * math.copysign(1.0, fx)
                        fx = _safe_f(f, x)
        
        if stall_count > 5:
            # Stuck, try a larger perturbation
            for scale in [0.1, 1.0, 10.0]:
                x_pert = x + scale * math.copysign(1.0, fx)
                fx_pert = _safe_f(f, x_pert)
                if math.isfinite(fx_pert) and abs(fx_pert) < best_fx:
                    x, fx = x_pert, fx_pert
                    stall_count = 0
                    break
    
    if abs(fx) <= 1e-8:
        return x
    
    # Fallback: scan a range around x0 and current x
    if time.monotonic() < deadline - 5.0:
        candidates = [x0, x, best_x]
        for cx in candidates:
            if not math.isfinite(cx):
                continue
            for r in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
                for sign in [1.0, -1.0]:
                    if time.monotonic() > deadline:
                        break
                    x_try = cx + sign * r
                    fx_try = _safe_f(f, x_try)
                    if math.isfinite(fx_try):
                        if abs(fx_try) <= 1e-8:
                            return x_try
                        # Try Newton from this point
                        x_n, fx_n = x_try, fx_try
                        for _ in range(15):
                            if time.monotonic() > deadline:
                                break
                            if abs(fx_n) <= 1e-8:
                                return x_n
                            x_n, fx_n, conv, fail = _newton_step(f, x_n, fx_n, h_base)
                            if conv:
                                return x_n
                            if fail:
                                break
                            if abs(fx_n) < abs(fx_try):
                                x_try, fx_try = x_n, fx_n
                            else:
                                break
    
    # Return best found
    if abs(_safe_f(f, best_x)) <= 1e-8:
        return best_x
    if abs(_safe_f(f, x)) <= 1e-8:
        return x
    if best_fx < abs(fx):
        return best_x
    return x

# EVOLVE-BLOCK-END
