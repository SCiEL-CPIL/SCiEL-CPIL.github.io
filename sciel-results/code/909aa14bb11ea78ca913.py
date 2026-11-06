"""Root finding for ill-conditioned problems using a robust hybrid approach."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Returns x with |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a combination of secant, Newton (with analytic or FD derivative),
    and bisection fallback for robustness on ill-conditioned problems.
    """
    x = float(x0)
    max_evals = 500
    eval_count = 0
    
    def eval_f(val):
        nonlocal eval_count
        if eval_count >= max_evals:
            raise RuntimeError("Max evaluations exceeded")
        result = f(val)
        eval_count += 1
        return result
    
    # Initial evaluation
    fx = eval_f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # Try to get an analytic derivative if available, otherwise use finite differences
    # We'll use a robust secant + Newton hybrid with bisection fallback
    
    # Secant method needs two points
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    fx1 = eval_f(x1)
    if abs(fx1) <= 1e-8:
        return x1
    
    # We'll maintain a bracket if possible, otherwise use secant/Newton
    # First, try to establish a bracket by expanding
    
    # Try to find a bracket [a, b] where f(a) and f(b) have opposite signs
    a = x
    fa = fx
    b = x1
    fb = fx1
    
    # Expand bracket if no sign change
    step = h
    for _ in range(20):
        if fa * fb < 0:
            break
        # Expand in the direction where |f| is decreasing, or just expand both sides
        if abs(fa) < abs(fb):
            # Expand b further
            b = b + step
        else:
            # Expand a further in the other direction
            a = a - step
        step *= 2
        # Check bounds
        if abs(a) > 1e10 or abs(b) > 1e10:
            break
        fa_new = eval_f(a)
        fb_new = eval_f(b)
        fa = fa_new
        fb = fb_new
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b
        if fa * fb < 0:
            break
    
    # Now use a hybrid method:
    # - If we have a bracket, use bisection + secant (Illinois or Brent-like)
    # - If not, use secant method with safeguards
    
    if fa * fb < 0:
        # We have a bracket [a, b]
        # Use a combination of bisection and secant (similar to Brent's method)
        # Keep track of the two most recent points
        a, fa = a, fa
        b, fb = b, fb
        c, fc = a, fa  # c is the previous b
        
        for _ in range(max_evals - eval_count):
            if abs(fb) <= 1e-8:
                return b
            if abs(fa) <= 1e-8:
                return a
            
            # Check if we've converged
            if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
                return b if abs(fb) < abs(fa) else a
            
            # Secant step
            if fa != fb:
                x_secant = b - fb * (b - a) / (fb - fa)
            else:
                x_secant = (a + b) / 2
            
            # Bisection step
            x_bisect = (a + b) / 2
            
            # Decide which step to take (Brent's logic)
            use_secant = False
            if abs(x_secant - x_bisect) < 1e-12:
                use_secant = True
            elif abs(x_secant - b) < abs(x_bisect - b):
                use_secant = True
            
            if use_secant:
                x_new = x_secant
            else:
                x_new = x_bisect
            
            # Ensure x_new is in [a, b]
            if x_new <= a or x_new >= b:
                x_new = x_bisect
            
            fx_new = eval_f(x_new)
            if abs(fx_new) <= 1e-8:
                return x_new
            
            # Update bracket
            if fa * fx_new < 0:
                b, fb = x_new, fx_new
            else:
                a, fa = x_new, fx_new
            
            # Convergence check
            if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
                return b if abs(fb) < abs(fa) else a
    else:
        # No bracket found; use secant method with safeguards
        # Keep two most recent points
        x0_sec = x
        f0_sec = fx
        x1_sec = x1
        f1_sec = fx1
        
        for _ in range(max_evals - eval_count):
            if abs(f1_sec) <= 1e-8:
                return x1_sec
            if abs(f0_sec) <= 1e-8:
                return x0_sec
            
            if f0_sec == f1_sec:
                # Degenerate case, perturb
                h2 = 1e-6 * max(1.0, abs(x1_sec))
                x1_sec = x1_sec + h2
                f1_sec = eval_f(x1_sec)
                continue
            
            x_new = x1_sec - f1_sec * (x1_sec - x0_sec) / (f1_sec - f0_sec)
            
            # Safeguard: if the step is too large or too small, use a smaller step
            step = x_new - x1_sec
            if abs(step) > 10 * max(1.0, abs(x1_sec)):
                x_new = x1_sec + 0.5 * step
            elif abs(step) < 1e-14 * max(1.0, abs(x1_sec)):
                break
            
            f_new = eval_f(x_new)
            if abs(f_new) <= 1e-8:
                return x_new
            
            # Update points
            x0_sec = x1_sec
            f0_sec = f1_sec
            x1_sec = x_new
            f1_sec = f_new
            
            # Convergence check
            if abs(x_new - x0_sec) < 1e-12 * max(1.0, abs(x0_sec), abs(x1_sec)):
                break
    
    # Return the best point found
    # Evaluate both candidate points and return the one with smaller |f|
    candidates = []
    if 'x' in dir():
        candidates.append(x)
    if 'x1' in dir():
        candidates.append(x1)
    if 'x0_sec' in dir():
        candidates.append(x0_sec)
    if 'x1_sec' in dir():
        candidates.append(x1_sec)
    if 'a' in dir() and 'fa' in dir():
        candidates.append(a)
    if 'b' in dir() and 'fb' in dir():
        candidates.append(b)
    
    if not candidates:
        return x
    
    # Find the candidate with the smallest |f|
    best_x = candidates[0]
    best_f = abs(eval_f(best_x))
    for cx in candidates[1:]:
        cf = abs(eval_f(cx))
        if cf < best_f:
            best_f = cf
            best_x = cx
    
    return best_x

# EVOLVE-BLOCK-END
