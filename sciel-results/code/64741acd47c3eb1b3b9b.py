import time

# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Find root x of f(x) such that |f(x)| <= 1e-8.
    Uses secant method with bisection fallback for robustness.
    """
    start_time = time.time()
    deadline = start_time + 10.0
    
    x_prev = float(x0)
    f_prev = f(x_prev)
    if abs(f_prev) <= 1e-10:
        return x_prev
    
    # Initialize second point with adaptive step
    step = 1e-3 * max(1.0, abs(x0))
    x_curr = x_prev + step
    f_curr = f(x_curr)
    
    if abs(f_curr) <= 1e-10:
        return x_curr
        
    best_x = x_prev if abs(f_prev) < abs(f_curr) else x_curr
    best_f = min(abs(f_prev), abs(f_curr))
    
    # Track bracket for bisection fallback
    # We want a bracket [a, b] where f(a) and f(b) have opposite signs
    a, b = None, None
    if f_prev * f_curr < 0:
        a, b = x_prev, x_curr
        fa, fb = f_prev, f_curr
    else:
        # Try to find a bracket by expanding
        for expand in range(20):
            new_step = step * (2 ** (expand + 1))
            x_test = x_prev + new_step
            f_test = f(x_test)
            if not math.isfinite(f_test):
                break
            if f_prev * f_test < 0:
                a, b = x_prev, x_test
                fa, fb = f_prev, f_test
                break
            if abs(f_test) < abs(f_curr):
                x_curr, f_curr = x_test, f_test
                best_x = x_curr
                best_f = abs(f_curr)
            if abs(f_test) <= 1e-10:
                return x_test
            if abs(f_test) < best_f:
                best_x = x_test
                best_f = abs(f_test)
    
    # Secant method with bisection fallback
    max_iter = 80
    stall_count = 0
    prev_best_f = best_f
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        # Check if we should switch to bisection
        use_bisection = False
        if a is not None and b is not None:
            # Check if secant step is going outside bracket or stalling
            denom = f_curr - f_prev
            if abs(denom) < 1e-15:
                use_bisection = True
            else:
                x_secant = x_curr - f_curr * (x_curr - x_prev) / denom
                # If secant step goes far outside bracket, use bisection
                if a is not None and b is not None:
                    bracket_size = abs(b - a)
                    if abs(x_secant - x_curr) > 2 * bracket_size:
                        use_bisection = True
        else:
            # No bracket, check for divergence
            denom = f_curr - f_prev
            if abs(denom) < 1e-15:
                use_bisection = True
            else:
                x_secant = x_curr - f_curr * (x_curr - x_prev) / denom
                if not math.isfinite(x_secant) or abs(x_secant - x_curr) > 100 * max(1.0, abs(x0)):
                    use_bisection = True
        
        if use_bisection:
            # Bisection / Illinois method
            if a is None or b is None:
                # Can't do bisection without bracket, break
                break
            # Illinois bisection: move the wider endpoint
            if abs(fa) < abs(fb):
                a = (a + b) / 2
                fa = f(a)
            else:
                b = (a + b) / 2
                fb = f(b)
            
            if not math.isfinite(fa) or not math.isfinite(fb):
                break
                
            if abs(fa) < best_f:
                best_x = a
                best_f = abs(fa)
            if abs(fb) < best_f:
                best_x = b
                best_f = abs(fb)
                
            if abs(fa) <= 1e-10:
                return a
            if abs(fb) <= 1e-10:
                return b
                
            # Check for stall
            if best_f >= prev_best_f * 0.99:
                stall_count += 1
            else:
                stall_count = 0
            prev_best_f = best_f
            
            if stall_count > 10:
                break
        else:
            # Secant step
            denom = f_curr - f_prev
            if abs(denom) < 1e-15:
                break
            x_next = x_curr - f_curr * (x_curr - x_prev) / denom
            f_next = f(x_next)
            
            if not math.isfinite(f_next) or not math.isfinite(x_next):
                break
                
            if abs(f_next) < best_f:
                best_x = x_next
                best_f = abs(f_next)
            
            if abs(f_next) <= 1e-10:
                return x_next
            
            # Update bracket if we cross zero
            if f_prev * f_next < 0:
                a, b = x_prev, x_next
                fa, fb = f_prev, f_next
            elif f_curr * f_next < 0:
                a, b = x_curr, x_next
                fa, fb = f_curr, f_next
                
            # Check for stall
            if best_f >= prev_best_f * 0.99:
                stall_count += 1
            else:
                stall_count = 0
            prev_best_f = best_f
            
            if stall_count > 15:
                # Try to establish a bracket
                if a is None or b is None:
                    # Expand around current best
                    for expand in range(10):
                        s = step * (2 ** (expand + 1))
                        x_t1 = best_x + s
                        x_t2 = best_x - s
                        f_t1 = f(x_t1)
                        f_t2 = f(x_t2)
                        if math.isfinite(f_t1) and math.isfinite(f_t2):
                            if f(best_x) * f_t1 < 0:
                                a, b = best_x, x_t1
                                fa, fb = f(best_x), f_t1
                                break
                            if f(best_x) * f_t2 < 0:
                                a, b = x_t2, best_x
                                fa, fb = f_t2, f(best_x)
                                break
                    if a is None or b is None:
                        break
                stall_count = 0
            
            # Shift
            x_prev = x_curr
            f_prev = f_curr
            x_curr = x_next
            f_curr = f_next
        
    return best_x
# EVOLVE-BLOCK-END
