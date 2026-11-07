"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Hybrid root finder: secant + bisection with bracketing and Newton refinement.
    Designed for smooth polynomial equations.
    """
    deadline = time.monotonic() + 145.0  # safety margin before 150s limit
    x0 = float(x0)
    
    def call(x):
        """Call f with time guard."""
        if time.monotonic() >= deadline:
            raise TimeoutError()
        return f(x)
    
    # Phase 1: Try to find a bracket around x0
    # Evaluate at x0
    fx0 = call(x0)
    if abs(fx0) <= 1e-8:
        return x0
    
    # Try to find a sign change by expanding interval
    a = x0
    fa = fx0
    b = None
    fb = None
    
    # Try several offsets to find a bracket
    step = max(1.0, abs(x0) * 0.5)
    for direction in [-1, 1]:
        for i in range(1, 20):
            x_try = x0 + direction * step * (2 ** i)
            if not math.isfinite(x_try):
                break
            try:
                fx_try = call(x_try)
            except:
                break
            if not math.isfinite(fx_try):
                break
            if fa * fx_try <= 0:
                if direction == -1:
                    a, fa = x_try, fx_try
                    b, fb = x0, fx0
                else:
                    a, fa = x0, fx0
                    b, fb = x_try, fx_try
                break
            if abs(fx_try) <= 1e-8:
                return x_try
        if b is not None:
            break
    
    # If we found a bracket, use hybrid bisection/secant
    if b is not None and a is not None:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        # Hybrid method: use secant step if it stays in bracket, else bisection
        x_prev = a
        f_prev = fa
        x_curr = b
        f_curr = fb
        
        for _ in range(100):
            if time.monotonic() >= deadline:
                break
            if abs(f_curr) <= 1e-8:
                return x_curr
            if abs(f_prev) <= 1e-8:
                return x_prev
            
            # Secant step
            if f_curr != f_prev:
                x_secant = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            else:
                x_secant = (x_curr + x_prev) / 2
            
            # Check if secant step is within bracket
            if a < x_secant < b:
                x_next = x_secant
            else:
                # Bisection
                x_next = (a + b) / 2
            
            f_next = call(x_next)
            if not math.isfinite(f_next):
                # Fall back to bisection
                x_next = (a + b) / 2
                f_next = call(x_next)
            
            if abs(f_next) <= 1e-8:
                return x_next
            
            # Update bracket
            if fa * f_next <= 0:
                b = x_next
                fb = f_next
            else:
                a = x_next
                fa = f_next
            
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_next, f_next
            
            # Convergence check
            if abs(b - a) < 1e-12 * max(1.0, abs(x_curr)):
                break
        
        # Final polish: if we're close, do a few more secant/Newton iterations for higher precision
        # This helps when the bracket has converged but |f(x)| is still slightly above 1e-8
        if abs(f_curr) <= 1e-6:
            # Use secant with the two most recent points for a few more iterations
            for _ in range(10):
                if time.monotonic() >= deadline:
                    break
                if abs(f_curr) <= 1e-8:
                    return x_curr
                if f_curr != f_prev:
                    x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
                    if math.isfinite(x_new):
                        f_new = call(x_new)
                        if math.isfinite(f_new):
                            if abs(f_new) <= 1e-8:
                                return x_new
                            x_prev, f_prev = x_curr, f_curr
                            x_curr, f_curr = x_new, f_new
                        if abs(f_new) < abs(f_curr):
                            x_curr, f_curr = x_new, f_new
                else:
                    break
        
        # Return the point with smallest |f|
        candidates = [a, b, x_curr]
        best_x = x_curr
        best_f = abs(f_curr)
        for xc in candidates:
            if not math.isfinite(xc):
                continue
            try:
                fc = call(xc)
            except:
                continue
            if math.isfinite(fc) and abs(fc) < best_f:
                best_x = xc
                best_f = abs(fc)
        return best_x
    
    # Phase 2: No bracket found. Use Newton-like method with finite differences
    # but with safeguards
    x = x0
    fx = fx0
    best_x = x0
    best_fx = abs(fx)
    
    for _ in range(60):
        if time.monotonic() >= deadline:
            break
        if abs(fx) <= 1e-8:
            return x
        if abs(fx) < best_fx:
            best_x = x
            best_fx = abs(fx)
        
        h = 1e-7 * max(1.0, abs(x))
        try:
            fxh = call(x + h)
        except:
            break
        if not math.isfinite(fxh):
            break
        
        d = (fxh - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        
        x_new = x - fx / d
        
        # Damping if step is too large
        if not math.isfinite(x_new):
            break
        step_size = abs(x_new - x)
        if step_size > 10 * max(1.0, abs(x)):
            x_new = x + math.copysign(10 * max(1.0, abs(x)), x_new - x)
        
        try:
            fx_new = call(x_new)
        except:
            break
        if not math.isfinite(fx_new):
            break
        
        # If new point is worse, try smaller step
        if abs(fx_new) > 2 * abs(fx):
            x_new = x - 0.5 * fx / d
            try:
                fx_new = call(x_new)
            except:
                break
            if not math.isfinite(fx_new):
                break
        
        x = x_new
        fx = fx_new
    
    # Return best found
    if abs(fx) < best_fx:
        best_x = x
        best_fx = abs(fx)
    
    return best_x
# EVOLVE-BLOCK-END
