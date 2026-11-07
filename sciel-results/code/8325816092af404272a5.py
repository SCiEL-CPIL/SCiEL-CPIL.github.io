"""
Root finder using Unified Inverse Quadratic Interpolation (IQI) with dynamic bracketing.
Strategy:
1. Instead of a separate grid scan phase, we maintain a small history of points.
2. We expand the search range adaptively (geometrically) until a bracket [a, b] is found 
   where f(a) and f(b) have opposite signs.
3. Once a bracket is established, we switch to a hybrid IQI/Secant/Bisection loop.
   - IQI uses the last 3 points to find the next estimate.
   - Secant is used if IQI fails or is unstable.
   - Bisection is used as a safeguard to guarantee convergence and maintain the bracket.
4. This avoids the overhead of scipy.optimize.brentq's internal state management 
   and the cost of a dense grid scan, as the expansion is minimal (only enough to find a sign change).
"""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8.
    f: callable, takes float, returns float
    x0: float starting point
    returns: float x with |f(x)| <= 1e-8
    """
    x0 = float(x0)
    
    # Deadline guard: 120s limit, use 115s to be safe
    deadline = time.time() + 115.0
    
    # 1. Initial evaluation
    fx0 = f(x0)
    if not math.isfinite(fx0):
        # If f(x0) is bad, we need to move. Try a small step.
        x0 += 1.0
        fx0 = f(x0)
        if not math.isfinite(fx0):
            # Last resort: return x0, likely to fail but avoids exception
            return x0
            
    if abs(fx0) <= 1e-8:
        return x0

    # 2. Adaptive Bracketing Phase
    # We expand from x0 to find [a, b] with opposite signs.
    # We keep track of the best point (min |f|) in case we fail to bracket.
    best_x = x0
    best_fx = fx0
    
    # We will search in + direction and - direction.
    # To minimize calls, we expand geometrically.
    # Let's maintain two "frontiers": one moving right, one moving left.
    
    # State: (x, fx)
    # We start with x0.
    # Let's try to find a sign change with x0.
    
    # Directions: +1, -1
    # Step sizes: 1, 2, 4, 8...
    
    # We need to find a pair (x_left, x_right) such that f(x_left)*f(x_right) < 0.
    # Initially, we don't have a bracket.
    
    # Let's define a helper to evaluate and update best
    def eval_and_update(x, fx):
        nonlocal best_x, best_fx
        if math.isfinite(fx) and abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx
        return fx

    # We have x0, fx0.
    # Let's expand to the right first, then left, interleaved.
    # Actually, a more robust way for polynomials:
    # Expand a window [x0 - R, x0 + R] and check if signs at boundaries differ.
    # But we might miss a root if both boundaries have same sign as x0 (even number of roots inside).
    # However, for "3 fixed smooth polynomial equations", usually simple roots or well-behaved.
    # Let's try to find ANY sign change between any two points in our expanding set.
    
    # Set of points visited: {x0: fx0}
    points = {x0: fx0}
    
    # Expansion radii
    radii = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0, 100000.0]
    
    bracket_a = None
    bracket_b = None
    bracket_fa = None
    bracket_fb = None
    
    # We will check new points against ALL previous points to find a bracket.
    # This is O(N^2) but N is small (max ~40 points).
    
    for r in radii:
        if time.time() > deadline - 10.0:
            break
            
        # Check x0 + r
        xp = x0 + r
        fxp = f(xp)
        if math.isfinite(fxp):
            eval_and_update(xp, fxp)
            points[xp] = fxp
            # Check for bracket with any existing point
            for x_prev, fx_prev in points.items():
                if x_prev == xp: continue
                if fx_prev * fxp < 0:
                    bracket_a = min(x_prev, xp)
                    bracket_b = max(x_prev, xp)
                    bracket_fa = f(bracket_a)
                    bracket_fb = f(bracket_b)
                    break
            if bracket_a is not None:
                break
        
        # Check x0 - r
        xm = x0 - r
        fxm = f(xm)
        if math.isfinite(fxm):
            eval_and_update(xm, fxm)
            points[xm] = fxm
            # Check for bracket with any existing point
            for x_prev, fx_prev in points.items():
                if x_prev == xm: continue
                if fx_prev * fxm < 0:
                    bracket_a = min(x_prev, xm)
                    bracket_b = max(x_prev, xm)
                    bracket_fa = f(bracket_a)
                    bracket_fb = f(bracket_b)
                    break
            if bracket_a is not None:
                break
        
        if bracket_a is not None:
            break

    # If no bracket found, return best_x
    if bracket_a is None:
        # Maybe we can do a local Newton step if best_fx is small?
        if abs(best_fx) < 1e-4:
            x = best_x
            fx = best_fx
            for _ in range(10):
                h = 1e-7 * max(1.0, abs(x))
                d = (f(x + h) - fx) / h
                if d == 0 or not math.isfinite(d):
                    break
                x_new = x - fx / d
                if not math.isfinite(x_new):
                    break
                fx_new = f(x_new)
                if abs(fx_new) < abs(fx):
                    x, fx = x_new, fx_new
                else:
                    break
            if abs(fx) <= 1e-8:
                return x
        return best_x

    # 3. Refinement Phase: Hybrid IQI / Secant / Bisection
    # We have bracket [bracket_a, bracket_b] with f(a), f(b) opposite signs.
    # We want to converge to the root.
    
    a = bracket_a
    b = bracket_b
    fa = bracket_fa
    fb = bracket_fb
    
    # Ensure |fa| <= |fb| to use the one with smaller magnitude as the "last" point for interpolation?
    # Standard Brent's method logic:
    # c is the previous b.
    # We maintain a, b, c such that f(a) and f(b) have opposite signs, and |f(b)| <= |f(c)|.
    
    # Initialize c = a
    c = a
    fc = fa
    
    # History for IQI: We need 3 points for quadratic interpolation.
    # Let's store the last few points (x, fx) to use for IQI.
    # Points: (a, fa), (b, fb), (c, fc) - note c==a initially.
    # Actually, let's just use a list of the last 3 distinct points.
    
    history = [(a, fa), (b, fb)]
    
    max_iter = 100
    for i in range(max_iter):
        if time.time() > deadline - 5.0:
            break
            
        if abs(fb) <= 1e-8:
            return b
        if abs(fa) <= 1e-8:
            return a
            
        # Convergence check on interval
        if abs(b - a) < 1e-12 * max(1.0, abs(b)):
            return b if abs(fb) < abs(fa) else a
            
        # Determine next point x_new
        
        # Option 1: Inverse Quadratic Interpolation (IQI)
        # Requires 3 points. We have at least 2 (a, b). 
        # If we have 3 distinct points in history, use IQI.
        x_new = None
        
        if len(history) >= 3:
            # Get last 3 points
            p1, p2, p3 = history[-3], history[-2], history[-1]
            x1, f1 = p1
            x2, f2 = p2
            x3, f3 = p3
            
            # Check for distinct x values
            if x1 != x2 and x2 != x3 and x1 != x3:
                # Lagrange interpolation for x(f=0)
                # x = (x1*x2*f3/(f3-f1)*(f3-f2) + x2*x3*f1/(f1-f2)*(f1-f3) + x3*x1*f2/(f2-f3)*(f2-f1)) / ...
                # Wait, the formula for inverse quadratic interpolation:
                # x_new = x1 * (f2/(f2-f1)) * (f3/(f3-f1)) + 
                #         x2 * (f1/(f1-f2)) * (f3/(f3-f2)) +
                #         x3 * (f1/(f1-f3)) * (f2/(f2-f3))
                
                try:
                    term1 = x1 * (f2 / (f2 - f1)) * (f3 / (f3 - f1))
                    term2 = x2 * (f1 / (f1 - f2)) * (f3 / (f3 - f2))
                    term3 = x3 * (f1 / (f1 - f3)) * (f2 / (f2 - f3))
                    x_new = term1 + term2 + term3
                except ZeroDivisionError:
                    x_new = None
                    pass
                except:
                    x_new = None
                    pass

        # Option 2: Secant Method
        if x_new is None:
            # Use last two points
            if len(history) >= 2:
                x1, f1 = history[-2]
                x2, f2 = history[-1]
                if x1 != x2 and f1 != f2:
                    x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
                else:
                    x_new = None
            else:
                x_new = None
        
        # Option 3: Bisection (Safeguard)
        if x_new is None or not math.isfinite(x_new):
            x_new = (a + b) / 2.0
        else:
            # Safeguard: If x_new is outside [a, b] or too close to a or b, use bisection
            # Standard Brent's safeguard:
            # 1. x_new must be in (a, b)
            # 2. If b is close to c (previous b), use bisection
            # 3. If the step is too small, use bisection
            
            # Check if x_new is in bracket
            if not (a < x_new < b):
                x_new = (a + b) / 2.0
            else:
                # Additional safeguard: If the interval is very small, bisection is safer
                if abs(b - a) < 1e-8:
                    x_new = (a + b) / 2.0
                else:
                    # Check if we are making progress
                    # If x_new is very close to a or b, it might be unstable
                    pass

        # Evaluate f(x_new)
        fx_new = f(x_new)
        
        if not math.isfinite(fx_new):
            # If evaluation fails, shrink bracket? Just return best so far?
            # For now, break and return best
            break
            
        # Update bracket
        if fa * fx_new < 0:
            # Root is in [a, x_new]
            b = x_new
            fb = fx_new
        else:
            # Root is in [x_new, b]
            a = x_new
            fa = fx_new
            
        # Update history
        history.append((x_new, fx_new))
        # Keep only last 3 points to avoid memory growth and stale data
        if len(history) > 3:
            history.pop(0)
            
        # Update best
        if abs(fx_new) < abs(best_fx):
            best_x = x_new
            best_fx = fx_new

    # Final check
    if abs(fb) <= 1e-8:
        return b
    if abs(fa) <= 1e-8:
        return a
    
    # Return the point with smallest |f|
    if abs(fb) < abs(fa):
        return b
    else:
        return a

# EVOLVE-BLOCK-END
