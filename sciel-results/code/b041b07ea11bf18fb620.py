"""Direct root-finder: IQI-primary with bracket-bisect fallback and Newton polish."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Find root of f starting near x0.
    Returns x with |f(x)| <= 1e-8 using at most 500 evaluations.
    """
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    max_evals = 500
    eval_count = 0
    best_x = float(x0)
    best_fx = None
    best_abs_fx = float('inf')

    def safe_f(x):
        nonlocal eval_count, best_x, best_fx, best_abs_fx
        if time.time() > deadline:
            raise TimeoutError()
        if eval_count >= max_evals:
            raise ValueError("max evals")
        eval_count += 1
        val = f(x)
        val = float(val)
        if math.isfinite(val):
            abs_val = abs(val)
            if abs_val < best_abs_fx:
                best_abs_fx = abs_val
                best_x = x
                best_fx = val
        return val

    x = float(x0)
    
    # First, evaluate at x0
    try:
        fx0 = safe_f(x)
    except (TimeoutError, ValueError):
        return best_x

    if abs(fx0) <= 1e-8:
        return x

    # Initialize with two points for secant/IQI
    h = 1e-6 * max(1.0, abs(x))
    x1 = x + h
    try:
        fx1 = safe_f(x1)
    except (TimeoutError, ValueError):
        return best_x

    if abs(fx1) <= 1e-8:
        return x1

    # Keep track of the three most recent points for IQI
    # (x0, fx0), (x1, fx1), and we'll add a third
    # Try to get a third point via secant step
    if fx0 != fx1:
        x2 = x1 - fx1 * (x1 - x) / (fx1 - fx0)
        # Damp if step is too large
        if abs(x2 - x1) > 10 * max(1.0, abs(x1)):
            x2 = x1 + 0.5 * (x - x1)
    else:
        x2 = x1 + h

    try:
        fx2 = safe_f(x2)
    except (TimeoutError, ValueError):
        return best_x

    if abs(fx2) <= 1e-8:
        return x2

    # Now we have three points: (x, fx0), (x1, fx1), (x2, fx2)
    # Sort them by x for IQI
    points = [(x, fx0), (x1, fx1), (x2, fx2)]
    points.sort(key=lambda p: p[0])

    # Check for bracket
    has_bracket = False
    bracket_a, bracket_b = None, None
    for i in range(len(points) - 1):
        if points[i][1] * points[i+1][1] < 0:
            has_bracket = True
            bracket_a, bracket_b = points[i][0], points[i+1][0]
            break

    # Main iteration loop
    # We'll maintain a history of the last 3 points for IQI
    history = points[:]  # list of (x, fx) tuples
    
    for iteration in range(450):
        if time.time() > deadline:
            break
        if eval_count >= max_evals:
            break
        
        # Try IQI first
        x_new = None
        if len(history) >= 3:
            # Sort by x
            hist_sorted = sorted(history, key=lambda p: p[0])
            x_a, fx_a = hist_sorted[0]
            x_b, fx_b = hist_sorted[1]
            x_c, fx_c = hist_sorted[2]
            
            # Check for duplicate x values
            if abs(x_a - x_b) > 1e-14 and abs(x_b - x_c) > 1e-14 and abs(x_a - x_c) > 1e-14:
                # Inverse quadratic interpolation
                # Fit parabola y = a*x^2 + b*x + c to the three points
                # Use the formula for the root of the quadratic
                # Or use the standard IQI formula
                
                # Standard IQI: find x such that the quadratic through (x_a,fx_a), (x_b,fx_b), (x_c,fx_c) is zero
                # Using Lagrange interpolation or the explicit formula
                
                # Explicit formula for IQI root:
                # x_new = x_c + (x_c - x_b)*(x_c - x_a) / (2*fx_c*(fx_c - fx_b)) * ... 
                # Actually, let's use the standard formula:
                
                # The quadratic through (x1,f1), (x2,f2), (x3,f3) is:
                # Q(x) = f1*(x-x2)(x-x3)/((x1-x2)(x1-x3)) + f2*(x-x1)(x-x3)/((x2-x1)(x2-x3)) + f3*(x-x1)(x-x2)/((x3-x1)(x3-x2))
                # Setting Q(x) = 0 and solving for x gives:
                
                denom1 = (x_a - x_b) * (x_a - x_c)
                denom2 = (x_b - x_a) * (x_b - x_c)
                denom3 = (x_c - x_a) * (x_c - x_b)
                
                if abs(denom1) > 1e-30 and abs(denom2) > 1e-30 and abs(denom3) > 1e-30:
                    # Q(x) = A*x^2 + B*x + C = 0
                    # A = f1/((x1-x2)(x1-x3)) + f2/((x2-x1)(x2-x3)) + f3/((x3-x1)(x3-x2))
                    # B = -f1*(x2+x3)/((x1-x2)(x1-x3)) - f2*(x1+x3)/((x2-x1)(x2-x3)) - f3*(x1+x2)/((x3-x1)(x3-x2))
                    # C = f1*x2*x3/((x1-x2)(x1-x3)) + f2*x1*x3/((x2-x1)(x2-x3)) + f3*x1*x2/((x3-x1)(x3-x2))
                    
                    A = fx_a / denom1 + fx_b / denom2 + fx_c / denom3
                    B = -fx_a * (x_b + x_c) / denom1 - fx_b * (x_a + x_c) / denom2 - fx_c * (x_a + x_b) / denom3
                    C = fx_a * x_b * x_c / denom1 + fx_b * x_a * x_c / denom2 + fx_c * x_a * x_b / denom3
                    
                    if abs(A) > 1e-30:
                        disc = B * B - 4 * A * C
                        if disc >= 0:
                            sqrt_disc = math.sqrt(disc)
                            # Two roots
                            root1 = (-B + sqrt_disc) / (2 * A)
                            root2 = (-B - sqrt_disc) / (2 * A)
                            # Choose the root that is closest to the most recent point (x_c)
                            # and is within a reasonable range
                            d1 = abs(root1 - x_c)
                            d2 = abs(root2 - x_c)
                            x_candidate = root1 if d1 < d2 else root2
                            
                            # Check if it's reasonable: not too far from the bracket or recent points
                            max_dist = 10 * max(1.0, abs(x_c), abs(x_b), abs(x_a))
                            if abs(x_candidate - x_c) < max_dist:
                                x_new = x_candidate
                    else:
                        # Linear case
                        if abs(B) > 1e-30:
                            x_new = -C / B
        
        # If IQI failed or produced nothing, use secant
        if x_new is None:
            # Use the two most recent points (by x order, the last two)
            if len(history) >= 2:
                hist_sorted = sorted(history, key=lambda p: p[0])
                x_p, fx_p = hist_sorted[-2]
                x_q, fx_q = hist_sorted[-1]
                if abs(fx_p - fx_q) > 1e-30:
                    x_new = x_q - fx_q * (x_q - x_p) / (fx_q - fx_p)
                    # Damp
                    if abs(x_new - x_q) > 10 * max(1.0, abs(x_q)):
                        x_new = x_q + 0.5 * (x_p - x_q)
                else:
                    x_new = 0.5 * (x_p + x_q)
            else:
                break
        
        # If we have a bracket, check if x_new is within it; if not, use bisection
        if has_bracket and bracket_a is not None and bracket_b is not None:
            # Update bracket from history
            hist_sorted = sorted(history, key=lambda p: p[0])
            for i in range(len(hist_sorted) - 1):
                if hist_sorted[i][1] * hist_sorted[i+1][1] < 0:
                    bracket_a, bracket_b = hist_sorted[i][0], hist_sorted[i+1][0]
                    break
            else:
                has_bracket = False
            
            if has_bracket:
                lo = min(bracket_a, bracket_b)
                hi = max(bracket_a, bracket_b)
                if not (lo < x_new < hi):
                    x_new = 0.5 * (lo + hi)
        
        try:
            fx_new = safe_f(x_new)
        except (TimeoutError, ValueError):
            break
        
        if abs(fx_new) <= 1e-8:
            return x_new
        
        # Update history: keep the 3 points with smallest |f|
        history.append((x_new, fx_new))
        history.sort(key=lambda p: abs(p[1]))
        history = history[:3]
        
        # Check convergence
        if len(history) >= 2:
            hist_sorted = sorted(history, key=lambda p: p[0])
            if abs(hist_sorted[-1][0] - hist_sorted[-2][0]) < 1e-14 * max(1.0, abs(hist_sorted[-1][0])):
                break

    return best_x
# EVOLVE-BLOCK-END
