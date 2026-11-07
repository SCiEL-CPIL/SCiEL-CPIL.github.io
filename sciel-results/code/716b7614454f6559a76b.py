"""
Combined strategy: Hybrid Newton-Illinois root finder.
1. Newton's method with central difference derivative for fast local convergence.
2. Adaptive bracketing (doubling step) if Newton fails.
3. Illinois method for robust refinement within the bracket.
"""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder for oscillatory equations.
    Strategy:
      1. Quick Newton attempt (up to 5 steps) to catch close roots efficiently.
      2. If not converged, find a bracket via expanding search (doubling steps).
      3. Use Illinois method to converge within tolerance.
    Budget: keep total calls well under 500.
    """
    start_time = time.time()
    TIME_LIMIT = 110.0  # safety margin under 120s
    TARGET = 1e-8
    MAX_CALLS = 480     # safety margin under 500
    
    call_count = 0
    def f_counted(x):
        nonlocal call_count
        call_count += 1
        try:
            val = f(x)
        except Exception:
            return float('nan')
        if not math.isfinite(val):
            return float('nan')
        return val
    
    x = float(x0)
    
    # Check initial point
    fx = f_counted(x)
    if abs(fx) <= TARGET:
        return x

    # Phase 1: Newton's method with central difference derivative
    # Central difference is more accurate for oscillatory functions to avoid jumping over roots
    # Cost: 2 calls per step (f(x+h), f(x-h)) + 1 for f(x_new) next iter = 3 calls/step effectively
    # But we reuse fx from previous step if we structure it right? 
    # Standard Newton: x_{n+1} = x_n - f(x_n)/f'(x_n)
    # We have fx. We need f'(x).
    # Let's do 3 iterations max.
    
    best_x = x
    best_f = abs(fx)
    
    for _ in range(3):
        if call_count >= MAX_CALLS - 10:
            break
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        h = 1e-5 * max(1.0, abs(x))
        fx_plus = f_counted(x + h)
        fx_minus = f_counted(x - h)
        
        if not math.isfinite(fx_plus) or not math.isfinite(fx_minus):
            break
            
        d = (fx_plus - fx_minus) / (2 * h)
        
        if d == 0 or not math.isfinite(d):
            break
            
        x_new = x - fx / d
        
        if not math.isfinite(x_new):
            break
            
        # Check for large jumps (divergence)
        if abs(x_new - x) > 10.0 * max(1.0, abs(x)):
            break
            
        x = x_new
        fx = f_counted(x)
        
        if not math.isfinite(fx):
            break
            
        if abs(fx) <= TARGET:
            return x
            
        if abs(fx) < best_f:
            best_f = fx
            best_x = x
            
        # Convergence check
        if abs(x_new - x) < 1e-12 * max(1.0, abs(x)):
            return x

    # If Newton didn't converge, we are at 'x' with 'fx'.
    # We need a bracket.
    # We will search for a sign change around 'x'.
    # Use doubling step size to cover range efficiently.
    
    a = x
    fa = fx
    found_bracket = False
    b = None
    fb = None
    
    step = 1e-3
    
    # Try expanding to the right first, then left
    for direction in [1, -1]:
        step = 1e-3
        for _ in range(60):  # 2^60 * 1e-3 is huge, but we stop at 100s or MAX_CALLS
            if call_count >= MAX_CALLS - 10:
                break
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            x_try = a + direction * step
            f_try = f_counted(x_try)
            
            if not math.isfinite(f_try):
                step *= 2.0
                continue
                
            if abs(f_try) <= TARGET:
                return x_try
                
            if fa * f_try < 0:
                # Found bracket [a, x_try]
                b = x_try
                fb = f_try
                found_bracket = True
                break
                
            step *= 2.0
            if step > 1e6:
                break
        
        if found_bracket:
            break
            
    if not found_bracket:
        # No bracket found. Return best guess.
        # Try a few more random-ish spots? No, just return best_x.
        return best_x

    # Ensure a < b for standard Illinois implementation
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
        
    # Phase 3: Illinois Method
    # Illinois is a modified bisection that converges faster than bisection 
    # and is more robust than secant for oscillatory functions.
    
    # Ensure we have valid finite values
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return best_x
        
    # Illinois algorithm
    # Keep a, b as the bracket.
    # fa, fb are function values at a, b.
    # We want to find x such that f(x) = 0.
    
    # To improve convergence, we can swap a and b if |fa| < |fb|? 
    # Standard Illinois:
    #   c = (a*fb - b*fa) / (fb - fa)  [Secant point]
    #   if c is not in (a, b), use midpoint.
    #   f(c)
    #   if f(a)*f(c) < 0: b = c, fb = fc; fa = fa/2 (Illinois modification: shrink the other side)
    #   else: a = c, fa = fc; fb = fb/2
    
    # Actually, a simpler and very effective variant is just using the secant point 
    # but falling back to bisection if it's outside. This is often called "Brent-like" 
    # without the full complexity. Given the call budget, let's use a clean Illinois.
    
    # Reset fa, fb to ensure they correspond to a, b
    # (They already do)
    
    for _ in range(100):
        if call_count >= MAX_CALLS:
            break
        if time.time() - start_time > TIME_LIMIT - 2:
            break
            
        # Compute secant point
        denom = fb - fa
        if abs(denom) < 1e-30:
            c = 0.5 * (a + b)
        else:
            c = b - fb * (b - a) / denom
            
        # If c is not strictly inside, use bisection
        if c <= a or c >= b:
            c = 0.5 * (a + b)
            
        fc = f_counted(c)
        
        if not math.isfinite(fc):
            # If we get a non-finite value, use bisection
            c = 0.5 * (a + b)
            fc = f_counted(c)
            if not math.isfinite(fc):
                break
                
        if abs(fc) <= TARGET:
            return c
            
        if abs(fc) < best_f:
            best_f = fc
            best_x = c
            
        # Update bracket
        if fa * fc < 0:
            # Root is in [a, c]
            b = c
            fb = fc
            # Illinois modification: halve the other side's function value to pull the secant point towards the midpoint
            # This ensures linear convergence with a rate of ~0.5 (like bisection) but usually faster
            fa = fa * 0.5
        else:
            # Root is in [c, b]
            a = c
            fa = fc
            fb = fb * 0.5
            
        # Convergence check on interval width
        if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
            break
            
    return best_x
# EVOLVE-BLOCK-END
