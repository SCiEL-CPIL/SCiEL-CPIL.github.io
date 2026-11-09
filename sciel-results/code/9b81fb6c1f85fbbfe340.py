"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s wall-clock limit
    MAX_CALLS = 450
    TOL = 1e-8
    STRICT_TOL = 1e-14
    
    calls = [0]
    
    def feval(x):
        if calls[0] >= MAX_CALLS:
            return None
        calls[0] += 1
        return f(x)
    
    def check_time():
        return time.time() < DEADLINE
    
    x = float(x0)
    
    # Initial evaluation
    fx = feval(x)
    if fx is None:
        return x
    if abs(fx) <= TOL:
        return x
    
    best_x = x
    best_fx = fx
    best_abs = abs(fx)
    
    # We'll use a hybrid approach:
    # 1. First try to establish a bracket using bisection-like expansion
    # 2. Then use cubic interpolation / inverse quadratic interpolation
    # 3. Fall back to secant/Newton if needed
    
    # Store points for interpolation: list of (x, f(x))
    points = [(x, fx)]
    
    # Try to find a sign change by expanding around x0
    a, b = None, None
    fa, fb = None, None
    
    # Check immediate neighbors for sign change
    step = max(1e-4, abs(x) * 1e-4)
    
    # Try several steps to find a bracket
    found_bracket = False
    for i in range(1, 20):
        if not check_time() or calls[0] >= MAX_CALLS - 10:
            break
        s = step * (2 ** i)
        x_left = x - s
        x_right = x + s
        
        fx_left = feval(x_left)
        if fx_left is None:
            break
        if abs(fx_left) < best_abs:
            best_x, best_fx, best_abs = x_left, fx_left, abs(fx_left)
        points.append((x_left, fx_left))
        
        if abs(fx_left) <= TOL:
            return x_left
        
        fx_right = feval(x_right)
        if fx_right is None:
            break
        if abs(fx_right) < best_abs:
            best_x, best_fx, best_abs = x_right, fx_right, abs(fx_right)
        points.append((x_right, fx_right))
        
        if abs(fx_right) <= TOL:
            return x_right
        
        # Check for sign change
        if fx_left * fx_right < 0:
            a, b = x_left, x_right
            fa, fb = fx_left, fx_right
            found_bracket = True
            break
    
    # If we have a bracket, use cubic interpolation within it
    if found_bracket and check_time() and calls[0] < MAX_CALLS:
        # We have a bracket [a, b] with fa*fb < 0
        # Use cubic interpolation with points we have
        # First, let's collect 4 points for cubic interpolation
        
        # We already have points. Let's use the most recent 4 distinct points
        # that are within or near the bracket
        
        # For cubic interpolation, we need 4 points
        # Let's use a, b, and two more points we've evaluated
        
        # Actually, let's use a more robust approach:
        # Use the bisection + interpolation hybrid (like Brent's method but with cubic)
        
        # Let's try cubic interpolation using 4 points
        # We'll use a, b, and the two points closest to the midpoint
        
        mid = (a + b) / 2.0
        fmid = feval(mid)
        if fmid is None:
            return best_x
        if abs(fmid) < best_abs:
            best_x, best_fx, best_abs = mid, fmid, abs(fmid)
        if abs(fmid) <= TOL:
            return mid
        points.append((mid, fmid))
        
        # Now we have 4 points: a, mid, b, and one more (x0 or another)
        # Let's use cubic interpolation on {a, mid, b, x} where x is our original point
        # Or better: use the 4 points that are most spread out within [a,b]
        
        # For simplicity, let's use the secant method with acceleration
        # but try cubic interpolation first
        
        # Cubic interpolation: given 4 points, find root of interpolating cubic
        # Points: (a,fa), (mid,fmid), (b,fb), and one more
        # Let's use (a,fa), (mid,fmid), (b,fb), and the point closest to but not equal to these
        
        # Actually, let's just use a robust iterative approach:
        # Alternate between bisection and cubic interpolation
        
        # Current bracket
        lo, hi = a, b
        flo, fhi = fa, fb
        
        # Ensure flo < 0 < fhi or vice versa
        if flo > 0:
            lo, hi = hi, lo
            flo, fhi = fhi, flo
        
        # We need 4 points for cubic. Let's collect them.
        # We have lo, hi, mid, and x (original)
        # Let's use these 4 points
        
        # Get 4 distinct x values
        xs = [lo, hi, mid, x]
        # Remove duplicates
        unique_xs = []
        for xi in xs:
            if not any(abs(xi - uj) < 1e-15 for uj in unique_xs):
                unique_xs.append(xi)
        
        if len(unique_xs) >= 4:
            # Use cubic interpolation
            xs4 = unique_xs[:4]
            fs4 = [feval(xi) if abs(xi - x) > 1e-15 else fx for xi in xs4]
            # Wait, we need to be careful not to double-count calls
            # Let's just use the points we already have
            
            # Actually, let's simplify: use secant + bisection hybrid
            pass
        
        # Let's use a simpler but effective approach:
        # Inverse quadratic interpolation with bisection fallback
        
        # We have points. Let's use the 3 most recent points for IQI
        # and fall back to bisection if IQI step is outside bracket
        
        # For now, let's use a modified secant method with bracketing
        
        # Iterative phase
        for _ in range(50):
            if not check_time() or calls[0] >= MAX_CALLS:
                break
            
            # Try cubic interpolation if we have 4 points in bracket
            # Otherwise use secant or bisection
            
            # Bisection point
            bisect_x = (lo + hi) / 2.0
            
            # Secant point using lo and hi
            if fhi != flo:
                secant_x = lo - flo * (hi - lo) / (fhi - flo)
            else:
                secant_x = bisect_x
            
            # Choose: if secant_x is within bracket, use it; else use bisection
            if lo < secant_x < hi:
                next_x = secant_x
            else:
                next_x = bisect_x
            
            fnext = feval(next_x)
            if fnext is None:
                break
            if abs(fnext) < best_abs:
                best_x, best_fx, best_abs = next_x, fnext, abs(fnext)
            if abs(fnext) <= TOL:
                return next_x
            
            # Update bracket
            if fnext * flo < 0:
                hi = next_x
                fhi = fnext
            else:
                lo = next_x
                flo = fnext
            
            # Check convergence
            if abs(hi - lo) < 1e-15:
                break
        
        return best_x
    
    # No bracket found; use Newton-like method with finite differences
    # but with better step control
    
    x = best_x
    for _ in range(40):
        if not check_time() or calls[0] >= MAX_CALLS:
            break
        
        fx = feval(x)
        if fx is None:
            break
        if abs(fx) < best_abs:
            best_x, best_fx, best_abs = x, fx, abs(fx)
        if abs(fx) <= TOL:
            return x
        
        h = 1e-7 * max(1.0, abs(x))
        d = (feval(x + h) - fx) / h
        if d is None or d == 0:
            break
        
        x_new = x - fx / d
        
        # Damping: if step is too large, reduce it
        step_size = abs(x_new - x)
        max_step = max(1.0, abs(x)) * 10.0
        if step_size > max_step:
            x_new = x + max_step * (1 if x_new > x else -1)
        
        x = x_new
    
    return best_x
# EVOLVE-BLOCK-END
