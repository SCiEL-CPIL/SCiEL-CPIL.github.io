"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Hybrid root finder: secant + bisection with bracketing and Newton refinement.
    Optimized to minimize function calls for smooth polynomial equations.
    """
    deadline = time.monotonic() + 145.0
    x0 = float(x0)
    
    call_count = [0]
    def call(x):
        """Call f with time guard and counting."""
        if time.monotonic() >= deadline:
            raise TimeoutError()
        call_count[0] += 1
        return f(x)
    
    # Evaluate at x0
    fx0 = call(x0)
    if abs(fx0) <= 1e-8:
        return x0
    
    # Phase 1: Find a bracket with minimal calls
    a = x0
    fa = fx0
    b = None
    fb = None
    
    # Try small steps first, then expand - for smooth polynomials roots are usually close
    step = max(0.1, abs(x0) * 0.1)
    for direction in [-1, 1]:
        for i in range(1, 15):
            x_try = x0 + direction * step * (2 ** i)
            if not math.isfinite(x_try):
                break
            try:
                fx_try = call(x_try)
            except:
                break
            if not math.isfinite(fx_try):
                break
            if abs(fx_try) <= 1e-8:
                return x_try
            if fa * fx_try <= 0:
                if direction == -1:
                    a, fa = x_try, fx_try
                    b, fb = x0, fx0
                else:
                    a, fa = x0, fx0
                    b, fb = x_try, fx_try
                break
        if b is not None:
            break
    
    # If we found a bracket, use hybrid bisection/secant
    if b is not None and a is not None:
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        x_prev = a
        f_prev = fa
        x_curr = b
        f_curr = fb
        
        # Track best point found
        best_x = x_curr
        best_f = abs(f_curr)
        if abs(fa) < best_f:
            best_x = a
            best_f = abs(fa)
        
        for _ in range(80):
            if time.monotonic() >= deadline:
                break
            if abs(f_curr) <= 1e-8:
                return x_curr
            if abs(f_prev) <= 1e-8:
                return x_prev
            
            # Track best
            if abs(f_curr) < best_f:
                best_x = x_curr
                best_f = abs(f_curr)
            
            # Secant step
            if f_curr != f_prev:
                x_secant = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            else:
                x_secant = (x_curr + x_prev) / 2
            
            # Check if secant step is within bracket
            if a < x_secant < b:
                x_next = x_secant
            else:
                x_next = (a + b) / 2
            
            f_next = call(x_next)
            if not math.isfinite(f_next):
                x_next = (a + b) / 2
                f_next = call(x_next)
            
            if abs(f_next) <= 1e-8:
                return x_next
            
            if abs(f_next) < best_f:
                best_x = x_next
                best_f = abs(f_next)
            
            # Update bracket
            if fa * f_next <= 0:
                b = x_next
                fb = f_next
            else:
                a = x_next
                fa = f_next
            
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_next, f_next
            
            # Convergence check - tighter threshold
            if abs(b - a) < 1e-10 * max(1.0, abs(x_curr)):
                break
        
        # Final polish only if close
        if best_f <= 1e-5:
            for _ in range(8):
                if time.monotonic() >= deadline:
                    break
                if best_f <= 1e-8:
                    return best_x
                if f_curr != f_prev:
                    x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
                    if math.isfinite(x_new):
                        f_new = call(x_new)
                        if math.isfinite(f_new):
                            if abs(f_new) < best_f:
                                best_x = x_new
                                best_f = abs(f_new)
                            if best_f <= 1e-8:
                                return best_x
                            x_prev, f_prev = x_curr, f_curr
                            x_curr, f_curr = x_new, f_new
                        else:
                            break
                else:
                    break
        
        return best_x
    
    # Phase 2: No bracket found. Use Newton-like method
    x = x0
    fx = fx0
    best_x = x0
    best_fx = abs(fx)
    
    for _ in range(50):
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
    
    if abs(fx) < best_fx:
        best_x = x
        best_fx = abs(fx)
    
    return best_x
# EVOLVE-BLOCK-END
