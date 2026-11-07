"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from scipy.optimize import brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 145.0  # 15s safety margin before 160s hard kill / 120s soft limit
    
    x = float(x0)
    
    # 1. Check if starting point is already a root
    fx = f(x)
    if not math.isfinite(fx):
        # If f(x0) is non-finite, try to find a nearby finite point
        # This is a rare edge case, but we handle it by scanning
        found = False
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]:
            for sign in [1, -1]:
                x_test = x + sign * dx
                fx_test = f(x_test)
                if math.isfinite(fx_test):
                    x = x_test
                    fx = fx_test
                    found = True
                    break
            if found:
                break
        if not found:
            return x # Return x0, likely a failure, but avoids crash
        if abs(fx) <= 1e-8:
            return x
    else:
        if abs(fx) <= 1e-8:
            return x

    # 2. Geometric Bracket Search
    # We want to find a, b such that f(a) and f(b) have opposite signs.
    # Start with a small step and expand geometrically.
    a = x
    fa = fx
    
    # Try expanding to the right first, then left if that fails
    # Actually, it's better to expand in both directions or just pick a direction
    # and expand. Let's try expanding to the right first.
    
    step = 1e-3
    b = None
    fb = None
    
    # Limit steps to avoid infinite loops or excessive calls
    # 50 steps of geometric expansion (1e-3 * 2^50 is huge)
    for i in range(60):
        if time.time() > deadline:
            # Time out: return best guess (x0 or last known)
            # We can't easily return "best" without tracking, so return x
            # But if we are in the middle of bracketing, x is just the last point checked
            # Actually, we should return the point with smallest |f| if we tracked it.
            # For simplicity, return x0 if we haven't found a bracket.
            # But wait, we might have found a good point.
            # Let's just return x0 to be safe, or the last x we evaluated?
            # The problem says "return x with |f(x)| <= 1e-8". If we fail, we return something.
            # Returning x0 is the default behavior on failure in many cases.
            return x0
        
        b = a + step
        fb = f(b)
        if not math.isfinite(fb):
            step *= 2
            continue
        
        if fa == 0:
            return a
        if fb == 0:
            return b
        
        if fa * fb < 0:
            # Bracket found!
            break
        
        step *= 2
    
    if b is None or not math.isfinite(fb) or fa * fb >= 0:
        # No bracket found to the right. Try to the left.
        step = 1e-3
        b = None
        fb = None
        for i in range(60):
            if time.time() > deadline:
                return x0
            
            b = a - step
            fb = f(b)
            if not math.isfinite(fb):
                step *= 2
                continue
            
            if fa == 0:
                return a
            if fb == 0:
                return b
            
            if fa * fb < 0:
                break
            
            step *= 2
    
    if b is None or not math.isfinite(fb) or fa * fb >= 0:
        # Still no bracket. This is a failure case for bracketing.
        # We could try a local minimization of |f(x)|, but that's complex.
        # Let's just return x0.
        return x0

    # 3. Brent's Method
    # Ensure a < b
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
    
    try:
        # xtol=1e-12 is tighter than 1e-8 requirement to ensure |f(x)| is small
        # rtol=1e-12
        root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
        
        # Verify the root
        f_root = f(root)
        if math.isfinite(f_root) and abs(f_root) <= 1e-8:
            return root
        else:
            # If brentq returned something weird, return x0
            return x0
            
    except Exception:
        # If brentq fails, return x0
        return x0
# EVOLVE-BLOCK-END
