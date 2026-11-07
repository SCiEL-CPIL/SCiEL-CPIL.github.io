"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust derivative-free root finder using hybrid bisection/secant method
    with bracketing, targeting smooth polynomial equations.
    """
    deadline = time.time() + 115.0  # safety margin before 120s limit
    x = float(x0)
    
    # Initial evaluation
    fx = f(x)
    if abs(fx) <= 1e-12:
        return x
    
    best_x = x
    best_fx = abs(fx)
    
    # Try to find a bracket with multiple step sizes
    a, b = None, None
    fa, fb = None, None
    h_base = max(1e-6, 1e-4 * abs(x)) if x != 0 else 1e-4
    for i in range(20):
        h = h_base * (2.0 ** i)
        x1 = x + h
        f1 = f(x1)
        if abs(f1) <= 1e-12:
            return x1
        if abs(f1) < best_fx:
            best_fx = abs(f1)
            best_x = x1
        
        x2 = x - h
        f2 = f(x2)
        if abs(f2) <= 1e-12:
            return x2
        if abs(f2) < best_fx:
            best_fx = abs(f2)
            best_x = x2
        
        # Check for sign change between x and x1
        if fx * f1 < 0:
            a, b = x, x1
            fa, fb = fx, f1
            break
        # Check for sign change between x and x2
        if fx * f2 < 0:
            a, b = x2, x
            fa, fb = f2, fx
            break
        # Check for sign change between x1 and x2
        if f1 * f2 < 0:
            a, b = x2, x1
            fa, fb = f2, f1
            break
        
        if time.time() > deadline:
            break
    
    # If we have a bracket, use bisection/secant hybrid
    if a is not None and b is not None:
        # Ensure |a| <= |b| in terms of function value for better tracking
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        for it in range(100):
            if time.time() > deadline:
                break
            
            # Try secant step
            if fa != fb:
                x_new = a - fa * (b - a) / (fb - fa)
            else:
                x_new = (a + b) / 2.0
            
            # Safeguard: keep within bracket
            if not math.isfinite(x_new) or x_new <= a or x_new >= b:
                x_new = (a + b) / 2.0
            
            f_new = f(x_new)
            if abs(f_new) <= 1e-12:
                return x_new
            if abs(f_new) < best_fx:
                best_fx = abs(f_new)
                best_x = x_new
            
            # Update bracket
            if fa * f_new <= 0:
                b, fb = x_new, f_new
            else:
                a, fa = x_new, f_new
            
            # Convergence check
            if abs(f_new) <= 1e-10:
                return x_new
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break
    
    # If no bracket found or after bracket convergence, try secant method
    # Use the two best points we have
    # Re-evaluate to get fresh values
    x_a = best_x
    f_a = f(x_a)
    if abs(f_a) <= 1e-12:
        return x_a
    
    # Get a second point
    h = 1e-4 * max(1.0, abs(x_a))
    x_b = x_a + h
    f_b = f(x_b)
    if abs(f_b) <= 1e-12:
        return x_b
    
    # Secant iterations
    for it in range(50):
        if time.time() > deadline:
            break
        
        if f_a == f_b:
            break
        x_new = x_a - f_a * (x_b - x_a) / (f_b - f_a)
        
        if not math.isfinite(x_new):
            break
        
        f_new = f(x_new)
        if abs(f_new) <= 1e-12:
            return x_new
        if abs(f_new) < best_fx:
            best_fx = abs(f_new)
            best_x = x_new
        
        # Update: keep the two points with smaller |f|
        if abs(f_new) < abs(f_b):
            x_b, f_b = x_new, f_new
        else:
            x_a, f_a = x_new, f_new
        
        if abs(f_new) <= 1e-10:
            return x_new
    
    # Final high-precision polish using Newton-like method
    if best_fx > 1e-8:
        xp = best_x
        fp = f(xp)
        for _ in range(20):
            if time.time() > deadline:
                break
            if abs(fp) <= 1e-10:
                return xp
            # Estimate derivative with central difference
            dx = 1e-7 * max(1.0, abs(xp))
            fp_plus = f(xp + dx)
            fp_minus = f(xp - dx)
            df = (fp_plus - fp_minus) / (2.0 * dx)
            if abs(df) < 1e-30:
                break
            x_new = xp - fp / df
            if not math.isfinite(x_new):
                break
            f_new = f(x_new)
            if abs(f_new) <= 1e-10:
                return x_new
            if abs(f_new) < best_fx:
                best_fx = abs(f_new)
                best_x = x_new
            xp = x_new
            fp = f_new
    
    return best_x
# EVOLVE-BLOCK-END
