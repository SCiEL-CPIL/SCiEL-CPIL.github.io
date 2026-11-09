import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    max_calls = 480
    call_count = 0
    best_x = float(x0)
    best_f = float('inf')

    def time_ok():
        return (time.time() - start_time) < deadline and call_count < max_calls

    def f_call(x):
        nonlocal call_count, best_x, best_f
        call_count += 1
        fx = f(x)
        if math.isfinite(fx) and abs(fx) < abs(best_f):
            best_f = fx
            best_x = x
        return fx

    x = float(x0)
    if not math.isfinite(x):
        return best_x

    fx = f_call(x)
    
    if not math.isfinite(fx):
        return best_x

    if abs(fx) <= 1e-8:
        return x

    # Phase 1: Find a bracket using a more aggressive search
    # Try multiple starting points and step sizes
    a = x
    fa = fx
    h = 1.0
    b = None
    fb = None
    
    # Try expanding with different step multipliers
    for _ in range(15):
        if not time_ok():
            break
        
        # Try expanding to right and left
        b_right = a + h
        fb_right = f_call(b_right)
        if math.isfinite(fb_right) and fa * fb_right < 0:
            b = b_right
            fb = fb_right
            break
        
        b_left = a - h
        fb_left = f_call(b_left)
        if math.isfinite(fb_left) and fa * fb_left < 0:
            b = b_left
            fb = fb_left
            break
        
        # If no sign change, expand further
        # Keep the point with smaller |f| as the new anchor
        if math.isfinite(fb_right) and abs(fb_right) < abs(fa):
            a = b_right
            fa = fb_right
        elif math.isfinite(fb_left) and abs(fb_left) < abs(fa):
            a = b_left
            fa = fb_left
            
        h *= 2.0

    if b is None:
        # Try a different starting point if initial bracketing failed
        # Try a few random-like offsets from x0
        offsets = [0.1, -0.1, 0.5, -0.5, 2.0, -2.0, 10.0, -10.0]
        for offset in offsets:
            if not time_ok():
                break
            
            x_try = x0 + offset * max(1.0, abs(x0))
            if not math.isfinite(x_try):
                continue
                
            fx_try = f_call(x_try)
            if not math.isfinite(fx_try):
                continue
                
            if abs(fx_try) <= 1e-8:
                return x_try
            
            # Try to bracket from this new point
            a_try = x_try
            fa_try = fx_try
            h_try = 0.1 * max(1.0, abs(x_try))
            
            for _ in range(8):
                if not time_ok():
                    break
                    
                b_right = a_try + h_try
                fb_right = f_call(b_right)
                if math.isfinite(fb_right) and fa_try * fb_right < 0:
                    a = a_try
                    fa = fa_try
                    b = b_right
                    fb = fb_right
                    break
                
                b_left = a_try - h_try
                fb_left = f_call(b_left)
                if math.isfinite(fb_left) and fa_try * fb_left < 0:
                    a = a_try
                    fa = fa_try
                    b = b_left
                    fb = fb_left
                    break
                    
                h_try *= 2.0
                
            if b is not None:
                break

    if b is not None:
        # Phase 2: Refine using Illinois algorithm
        # Illinois is robust and efficient for 1D root finding
        # We maintain [a, b] with f(a)*f(b) < 0
        # And update the endpoint with larger |f| towards the other
        
        # Ensure a < b for consistency, though not strictly required for Illinois
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        # Illinois algorithm loop
        for _ in range(100):
            if not time_ok():
                break
                
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
                
            if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
                # Return the midpoint or the endpoint with smaller |f|
                if abs(fa) < abs(fb):
                    return a
                else:
                    return b
            
            # Illinois step
            if abs(fa) > abs(fb):
                # Replace a with a new point closer to b
                # x = a - (fa / (fb - fa)) * (a - b)  => secant
                # Illinois uses: x = a - 0.5 * (fa / (fb - fa)) * (a - b)
                # Or simpler: x = a - fa * (b - a) / (2 * (fb - fa))
                if fb != fa:
                    x_new = a - fa * (b - a) / (2.0 * (fb - fa))
                else:
                    x_new = 0.5 * (a + b)
                
                if not math.isfinite(x_new) or not (a < x_new < b):
                    x_new = 0.5 * (a + b)
                    
                fx_new = f_call(x_new)
                if not math.isfinite(fx_new):
                    break
                if abs(fx_new) <= 1e-8:
                    return x_new
                    
                if fa * fx_new < 0:
                    b = x_new
                    fb = fx_new
                else:
                    a = x_new
                    fa = fx_new
            else:
                # Replace b with a new point closer to a
                if fb != fa:
                    x_new = b - fb * (a - b) / (2.0 * (fa - fb))
                else:
                    x_new = 0.5 * (a + b)
                    
                if not math.isfinite(x_new) or not (a < x_new < b):
                    x_new = 0.5 * (a + b)
                    
                fx_new = f_call(x_new)
                if not math.isfinite(fx_new):
                    break
                if abs(fx_new) <= 1e-8:
                    return x_new
                    
                if fb * fx_new < 0:
                    a = x_new
                    fa = fx_new
                else:
                    b = x_new
                    fb = fx_new
                    
        # Return best found in bracket
        if abs(fa) < abs(fb):
            return a
        else:
            return b

    else:
        # No bracket found. Fall back to Newton-Raphson with damping
        x = float(x0)
        fx = f_call(x)
        
        for _ in range(60):
            if not time_ok():
                break
            if abs(fx) <= 1e-8:
                return x
                
            h = 1e-6 * max(1.0, abs(x))
            fx_plus = f_call(x + h)
            if not math.isfinite(fx_plus):
                break
            d = (fx_plus - fx) / h
            if d == 0 or not math.isfinite(d):
                break
                
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
                
            # Damping
            if abs(x_new - x) > 10.0:
                x_new = x + 10.0 * math.copysign(1.0, x_new - x)
                
            x = x_new
            fx = f_call(x)
            
        return best_x

    return best_x
