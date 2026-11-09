"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin for 150s limit
    
    def time_up():
        return time.time() >= deadline

    x0 = float(x0)
    
    # Track best solution found
    best_x = x0
    best_f = f(x0)
    best_abs = abs(best_f)
    
    def update_best(x, fx):
        nonlocal best_x, best_f, best_abs
        ax = abs(fx)
        if ax < best_abs:
            best_x = x
            best_f = fx
            best_abs = ax

    # If already good enough
    if best_abs <= 1e-8:
        return best_x

    # Global Inverse Quadratic Interpolation (IQI) Strategy
    # This method fits a quadratic polynomial q(x) to (f(x), x) pairs.
    # It is derivative-free and has cubic convergence near the root.
    # We maintain a window of the last 3 points to fit the parabola.
    
    # Initial points: x0, x0+1, x0-1
    # We need to be careful if f is undefined or huge, but for the task 
    # "fixed local ill-conditioned test equations", we assume continuity 
    # and finite values in the neighborhood.
    
    points = []
    
    def add_point(x, fx):
        nonlocal points
        points.append((x, fx))
        # Keep only the last 3 points for the fit
        if len(points) > 3:
            points.pop(0)

    # Initialize with x0
    add_point(x0, best_f)
    
    # Add two more points to establish the initial fit
    # Use a small step to avoid immediate divergence if the function is steep
    h0 = 1e-3
    x1 = x0 + h0
    x2 = x0 - h0
    
    # Evaluate x1
    if not time_up():
        f1 = f(x1)
        update_best(x1, f1)
        add_point(x1, f1)
        if abs(f1) <= 1e-8:
            return x1
            
    # Evaluate x2
    if not time_up():
        f2 = f(x2)
        update_best(x2, f2)
        add_point(x2, f2)
        if abs(f2) <= 1e-8:
            return x2

    # If we still don't have 3 points (unlikely), fill with larger steps
    while len(points) < 3 and not time_up():
        h0 *= 10
        xn = x0 + h0
        fn = f(xn)
        update_best(xn, fn)
        add_point(xn, fn)
        if abs(fn) <= 1e-8:
            return xn

    # Main IQI Loop
    # We iterate until we find a root or run out of time/budget
    # Budget: 500 evals. We used ~3 so far.
    # Each step uses 1 eval.
    # We can do ~490 steps.
    
    for _ in range(490):
        if time_up():
            break
            
        if len(points) < 3:
            # Fallback to bisection-like expansion if we lost points
            # This shouldn't happen with the logic above, but safe
            h0 *= 2
            xn = x0 + h0
            fn = f(xn)
            update_best(xn, fn)
            add_point(xn, fn)
            if abs(fn) <= 1e-8:
                return xn
            continue

        # Extract the last 3 points: (x1, f1), (x2, f2), (x3, f3)
        (x1, f1), (x2, f2), (x3, f3) = points[-3], points[-2], points[-1]
        
        # We want to find x such that f(x) = 0.
        # Inverse Quadratic Interpolation fits x as a function of y=f(x).
        # x(y) = A*y^2 + B*y + C
        # We solve for A, B, C such that:
        # A*f1^2 + B*f1 + C = x1
        # A*f2^2 + B*f2 + C = x2
        # A*f3^2 + B*f3 + C = x3
        
        # This is a linear system in A, B, C.
        # Let's use Cramer's rule or Gaussian elimination for 3x3.
        # Or simpler: Lagrange interpolation for x(y).
        
        # x(0) = x1 * ( (0-f2)(0-f3) / ((f1-f2)(f1-f3)) ) + 
        #       x2 * ( (0-f1)(0-f3) / ((f2-f1)(f2-f3)) ) + 
        #       x3 * ( (0-f1)(0-f2) / ((f3-f1)(f3-f2)) )
        
        # Simplify terms:
        # Term 1: x1 * (f2*f3) / ((f1-f2)(f1-f3))
        # Term 2: x2 * (f1*f3) / ((f2-f1)(f2-f3))  -> note (f2-f1) = -(f1-f2)
        # Term 3: x3 * (f1*f2) / ((f3-f1)(f3-f2))  -> note (f3-f1) = -(f1-f3), (f3-f2) = -(f2-f3)
        
        # Let's check denominators for zero (duplicate function values).
        d12 = f1 - f2
        d13 = f1 - f3
        d23 = f2 - f3
        
        if abs(d12) < 1e-15 or abs(d13) < 1e-15 or abs(d23) < 1e-15:
            # Function values are too close, IQI is unstable.
            # Fallback to Secant Method using the two points with largest f difference?
            # Or just Bisection on the range of x's?
            # For ill-conditioned, f values might be very close to 0 but not distinct enough.
            # Let's try Secant with the two most recent distinct points.
            if abs(d23) > 1e-15:
                # Secant between x2 and x3
                xn = x3 - f3 * (x3 - x2) / (f3 - f2)
            elif abs(d13) > 1e-15:
                xn = x3 - f3 * (x3 - x1) / (f3 - f1)
            else:
                # All f values nearly identical. Just bisect the x range.
                xn = (x1 + x3) / 2
        else:
            # Calculate x(0) using Lagrange formula
            # x(0) = x1 * (f2*f3) / ((f1-f2)*(f1-f3)) + 
            #       x2 * (f1*f3) / ((f2-f1)*(f2-f3)) +
            #       x3 * (f1*f2) / ((f3-f1)*(f3-f2))
            
            # Note signs:
            # (f2-f1) = -d12
            # (f3-f1) = -d13
            # (f3-f2) = -d23
            
            # Term 1: x1 * f2*f3 / (d12 * d13)
            # Term 2: x2 * f1*f3 / (-d12 * d23)  -> -x2 * f1*f3 / (d12 * d23)
            # Term 3: x3 * f1*f2 / (-d13 * -d23) -> x3 * f1*f2 / (d13 * d23)
            
            # Let's compute carefully
            denom1 = d12 * d13
            denom2 = d12 * d23
            denom3 = d13 * d23
            
            if abs(denom1) < 1e-30 or abs(denom2) < 1e-30 or abs(denom3) < 1e-30:
                xn = (x1 + x3) / 2
            else:
                t1 = x1 * (f2 * f3) / denom1
                t2 = -x2 * (f1 * f3) / denom2
                t3 = x3 * (f1 * f2) / denom3
                xn = t1 + t2 + t3

        # Safety check: If xn is NaN or Inf, fallback to bisection
        if not math.isfinite(xn):
            xn = (x1 + x3) / 2
            
        # Evaluate
        fn = f(xn)
        update_best(xn, fn)
        
        if abs(fn) <= 1e-8:
            return xn
            
        # Add to history
        add_point(xn, fn)
        
        # Optional: If the step was very large or went far away, we might want to reset
        # But IQI is generally robust if the initial points bracket the root or are close.
        # For "local ill-conditioned", the root is likely near x0.
        # If we drift too far, we might miss it. 
        # However, the "best_x" tracking ensures we don't lose the best candidate found so far.
        
    return best_x
# EVOLVE-BLOCK-END
