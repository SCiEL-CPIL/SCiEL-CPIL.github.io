import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0
    
    def time_ok():
        return time.time() < deadline

    x = float(x0)
    try:
        fx = f(x)
    except Exception:
        return x
        
    if not math.isfinite(fx):
        return x
        
    if abs(fx) <= 1e-8:
        return x

    # 1. Try to find a bracket
    # Use small steps to find a sign change
    h = 1e-6 * max(1.0, abs(x))
    
    # Evaluate neighbors
    x_left = x - h
    x_right = x + h
    
    try:
        f_left = f(x_left)
    except Exception:
        f_left = None
        
    try:
        f_right = f(x_right)
    except Exception:
        f_right = None
        
    a, b = None, None
    fa, fb = None, None
    
    if f_left is not None and math.isfinite(f_left):
        if abs(f_left) <= 1e-8:
            return x_left
        if fx * f_left < 0:
            a, b = x_left, x
            fa, fb = f_left, fx
    if b is None and f_right is not None and math.isfinite(f_right):
        if abs(f_right) <= 1e-8:
            return x_right
        if fx * f_right < 0:
            a, b = x, x_right
            fa, fb = fx, f_right
            
    # If no bracket, try expanding
    if b is None:
        # Expand right
        step = h
        prev_x, prev_f = x, fx
        curr_x, curr_f = x + h, f_right if f_right is not None else f(x + h)
        
        for _ in range(20):
            if not time_ok(): break
            if prev_f * curr_f < 0:
                a, b = prev_x, curr_x
                fa, fb = prev_f, curr_f
                break
            step *= 2
            next_x = x + step
            try:
                next_f = f(next_x)
            except Exception:
                break
            if not math.isfinite(next_f):
                break
            if abs(next_f) <= 1e-8:
                return next_x
            prev_x, prev_f = curr_x, curr_f
            curr_x, curr_f = next_x, next_f
            
        if b is None:
            # Expand left
            step = h
            prev_x, prev_f = x, fx
            curr_x, curr_f = x - h, f_left if f_left is not None else f(x - h)
            
            for _ in range(20):
                if not time_ok(): break
                if prev_f * curr_f < 0:
                    a, b = curr_x, prev_x
                    fa, fb = curr_f, prev_f
                    break
                step *= 2
                next_x = x - step
                try:
                    next_f = f(next_x)
                except Exception:
                    break
                if not math.isfinite(next_f):
                    break
                if abs(next_f) <= 1e-8:
                    return next_x
                prev_x, prev_f = curr_x, curr_f
                curr_x, curr_f = next_x, next_f

    if b is not None:
        # We have a bracket [a, b]
        # Use IQI. We need a 3rd point. Use midpoint.
        c = 0.5 * (a + b)
        try:
            fc = f(c)
        except Exception:
            fc = None
            c = None
            
        if c is None or not math.isfinite(fc):
            # Fallback to bisection
            while time_ok() and (b - a) > 1e-12:
                mid = 0.5 * (a + b)
                fm = f(mid)
                if abs(fm) <= 1e-8:
                    return mid
                if fa * fm < 0:
                    b, fb = mid, fm
                else:
                    a, fa = mid, fm
            return 0.5 * (a + b)
            
        # IQI Loop
        # Points: (a, fa), (b, fb), (c, fc)
        # We want to find the root of the quadratic interpolant.
        
        for _ in range(50):
            if not time_ok(): break
            
            # Check if any point is a root
            if abs(fa) <= 1e-8: return a
            if abs(fb) <= 1e-8: return b
            if abs(fc) <= 1e-8: return c
            
            # Solve for root of quadratic through (a,fa), (b,fb), (c,fc)
            x1, y1 = a, fa
            x2, y2 = b, fb
            x3, y3 = c, fc
            
            d12 = x1 - x2
            d13 = x1 - x3
            d23 = x2 - x3
            
            if d12 == 0 or d13 == 0 or d23 == 0:
                # Degenerate, use bisection
                x_new = 0.5 * (a + b)
            else:
                denom1 = d12 * d13
                denom2 = (-d12) * d23
                denom3 = d13 * d23
                
                A = y1/denom1 + y2/denom2 + y3/denom3
                B = -y1*(x2+x3)/denom1 - y2*(x1+x3)/denom2 - y3*(x1+x2)/denom3
                C = y1*x2*x3/denom1 + y2*x1*x3/denom2 + y3*x1*x2/denom3
                
                if abs(A) < 1e-15:
                    if abs(B) > 1e-15:
                        x_new = -C / B
                    else:
                        x_new = 0.5 * (a + b)
                else:
                    disc = B*B - 4*A*C
                    if disc < 0:
                        x_new = 0.5 * (a + b)
                    else:
                        sqrt_disc = math.sqrt(disc)
                        x_r1 = (-B + sqrt_disc) / (2*A)
                        x_r2 = (-B - sqrt_disc) / (2*A)
                        
                        if a <= x_r1 <= b:
                            x_new = x_r1
                        elif a <= x_r2 <= b:
                            x_new = x_r2
                        else:
                            x_new = 0.5 * (a + b)
                            
            # Evaluate
            try:
                f_new = f(x_new)
            except Exception:
                f_new = None
                x_new = None
                
            if x_new is None or not math.isfinite(f_new):
                x_new = 0.5 * (a + b)
                f_new = f(x_new)
                
            if abs(f_new) <= 1e-8:
                return x_new
                
            # Update bracket
            if fa * f_new < 0:
                b, fb = x_new, f_new
            else:
                a, fa = x_new, f_new
                
            # Update 3rd point c to be the midpoint of the new bracket
            c = 0.5 * (a + b)
            try:
                fc = f(c)
            except Exception:
                fc = None
                c = None
            if c is None:
                break
                
    else:
        # No bracket found. Use Secant.
        # We have (x, fx) and (x+h, f_right)
        x1, y1 = x, fx
        x2, y2 = x + h, f_right if f_right is not None else f(x + h)
        
        for _ in range(50):
            if not time_ok(): break
            if abs(y1) <= 1e-8: return x1
            if abs(y2) <= 1e-8: return x2
            if y1 == y2: break
            
            x3 = x2 - y2 * (x2 - x1) / (y2 - y1)
            try:
                y3 = f(x3)
            except Exception:
                break
            if not math.isfinite(y3):
                break
            if abs(y3) <= 1e-8:
                return x3
                
            # Check for bracket formation
            if y1 * y3 < 0:
                # Switch to IQI with bracket [x1, x3]
                a, b = x1, x3
                fa, fb = y1, y3
                c = 0.5 * (a + b)
                try:
                    fc = f(c)
                except Exception:
                    fc = None
                if c is not None and fc is not None and math.isfinite(fc):
                    # Enter IQI loop (simplified, just return best guess if tight)
                    # For simplicity, let's just continue Secant for now to avoid code bloat
                    pass
            elif y2 * y3 < 0:
                # Bracket [x2, x3]
                pass
                
            x1, y1 = x2, y2
            x2, y3 = x3, y3
            
        return x1 if abs(y1) < abs(y2) else x2

    return x
# EVOLVE-BLOCK-END
