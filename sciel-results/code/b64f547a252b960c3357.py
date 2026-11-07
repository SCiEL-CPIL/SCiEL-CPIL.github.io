"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find a root of f near x0 using a combination of bracketing (bisection)
    and inverse quadratic interpolation (IQI) / secant steps.
    Returns x with |f(x)| <= 1e-8.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    x = float(x0)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # Try to find a bracket [a, b] where f(a) and f(b) have opposite signs
    # Start with small steps and expand
    a = x
    fa = fx
    h = 1e-4
    max_h = 1e6
    
    bracket_found = False
    b = None
    fb = None
    
    # Try expanding in both directions
    for direction in [1, -1]:
        h = 1e-4
        while h <= max_h:
            if time.time() > deadline:
                break
            x_try = x + direction * h
            fx_try = f(x_try)
            if abs(fx_try) <= 1e-8:
                return x_try
            if fa * fx_try < 0:
                if direction == 1:
                    a, b = x, x_try
                    fa, fb = fx, fx_try
                else:
                    a, b = x_try, x
                    fa, fb = fx_try, fx
                bracket_found = True
                break
            h *= 2.0
        if bracket_found:
            break
    
    if not bracket_found:
        # No bracket found, try secant/Newton-like approach
        # Use a few secant iterations
        x1 = x + 1e-4
        fx1 = f(x1)
        for _ in range(50):
            if time.time() > deadline:
                break
            if abs(fx1) <= 1e-8:
                return x1
            if abs(fx) <= 1e-8:
                return x
            if fx1 == fx:
                break
            x_new = x1 - fx1 * (x1 - x) / (fx1 - fx)
            x, fx = x1, fx1
            x1 = x_new
            fx1 = f(x1)
        return x if abs(fx) <= abs(fx1) else x1
    
    # Now we have a bracket [a, b] with fa*fb < 0
    # Use a combination of bisection and IQI/secant
    # Keep track of the best point (smallest |f|)
    best_x = a if abs(fa) < abs(fb) else b
    best_f = min(abs(fa), abs(fb))
    
    # We'll maintain three points for IQI: x1, x2, x3 (most recent first)
    # Initially we have a and b, need a third point
    # Use the midpoint as initial third point
    x_mid = (a + b) / 2.0
    fx_mid = f(x_mid)
    if abs(fx_mid) <= 1e-8:
        return x_mid
    if abs(fx_mid) < best_f:
        best_x = x_mid
        best_f = abs(fx_mid)
    
    # Determine which sub-interval to keep
    if fa * fx_mid < 0:
        b = x_mid
        fb = fx_mid
    else:
        a = x_mid
        fa = fx_mid
    
    # Now iterate using IQI when possible, otherwise bisection
    # Keep history of last 3 points
    history = [(b, fb), (a, fa)]  # (x, fx) pairs, most recent first
    
    for _ in range(200):
        if time.time() > deadline:
            break
        
        # Check if we already have a solution
        for xi, fxi in history:
            if abs(fxi) <= 1e-8:
                return xi
        
        # Try IQI if we have 3 distinct points
        if len(history) >= 3:
            x1, f1 = history[0]
            x2, f2 = history[1]
            x3, f3 = history[2]
            
            # Check if points are distinct enough
            if (abs(x1 - x2) > 1e-14 and abs(x2 - x3) > 1e-14 and abs(x1 - x3) > 1e-14):
                # Inverse quadratic interpolation
                # q(x) = q0 + q1*(x-x1) + q2*(x-x1)*(x-x2)
                # Solve q(xi) = fi for i=1,2,3
                # This gives:
                # q0 = f1
                # q1 = (f2-f1)/(x2-x1)
                # q2 = (f3-f1)/((x3-x1)*(x3-x2)) - q1/(x3-x2)
                # Then solve q(x) = 0: q0 + q1*(x-x1) + q2*(x-x1)*(x-x2) = 0
                # Let t = x - x1, then: q0 + q1*t + q2*t*(t + x1 - x2) = 0
                # q0 + q1*t + q2*t^2 + q2*(x1-x2)*t = 0
                # q2*t^2 + (q1 + q2*(x1-x2))*t + q0 = 0
                
                dx2 = x2 - x1
                dx3 = x3 - x1
                dx32 = x3 - x2
                
                if abs(dx2) > 1e-14 and abs(dx32) > 1e-14 and abs(dx3) > 1e-14:
                    q1 = (f2 - f1) / dx2
                    q2 = (f3 - f1) / (dx3 * dx32) - q1 / dx32
                    A = q2
                    B = q1 + q2 * (x1 - x2)
                    C = f1
                    
                    if abs(A) > 1e-14:
                        disc = B * B - 4 * A * C
                        if disc >= 0:
                            sqrt_disc = math.sqrt(disc)
                            # Two roots: (-B ± sqrt_disc) / (2*A)
                            # Choose the one that is within [a, b] and closest to x1
                            r1 = (-B + sqrt_disc) / (2 * A) + x1
                            r2 = (-B - sqrt_disc) / (2 * A) + x1
                            
                            # Prefer the root within the bracket
                            candidates = []
                            if a <= r1 <= b:
                                candidates.append(r1)
                            if a <= r2 <= b:
                                candidates.append(r2)
                            
                            if candidates:
                                # Choose the one with smaller |x - x1|
                                x_new = min(candidates, key=lambda xc: abs(xc - x1))
                            else:
                                # Use bisection
                                x_new = (a + b) / 2.0
                        else:
                            x_new = (a + b) / 2.0
                    else:
                        # Linear case
                        if abs(B) > 1e-14:
                            x_new = -C / B + x1
                            if not (a <= x_new <= b):
                                x_new = (a + b) / 2.0
                        else:
                            x_new = (a + b) / 2.0
                else:
                    x_new = (a + b) / 2.0
            else:
                x_new = (a + b) / 2.0
        else:
            # Use secant if we have 2 points
            if len(history) == 2:
                x1, f1 = history[0]
                x2, f2 = history[1]
                if abs(f1 - f2) > 1e-14:
                    x_new = x1 - f1 * (x1 - x2) / (f1 - f2)
                    if not (a <= x_new <= b):
                        x_new = (a + b) / 2.0
                else:
                    x_new = (a + b) / 2.0
            else:
                x_new = (a + b) / 2.0
        
        fx_new = f(x_new)
        if abs(fx_new) <= 1e-8:
            return x_new
        if abs(fx_new) < best_f:
            best_x = x_new
            best_f = abs(fx_new)
        
        # Update bracket
        if fa * fx_new < 0:
            b = x_new
            fb = fx_new
        else:
            a = x_new
            fa = fx_new
        
        # Update history
        history.insert(0, (x_new, fx_new))
        if len(history) > 3:
            history.pop()
    
    # Return the best point found
    for xi, fxi in history:
        if abs(fxi) <= 1e-8:
            return xi
    
    return best_x
# EVOLVE-BLOCK-END
