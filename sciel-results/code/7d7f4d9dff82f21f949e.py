import math
import time
from scipy.optimize import brentq

def find_root(f, x0):
    x0 = float(x0)
    fx0 = f(x0)
    
    # Check if already close enough
    if abs(fx0) <= 1e-8:
        return x0

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin for 3 equations + overhead

    # Try to find a bracket [a, b]
    # We expand the search interval exponentially
    step = 1.0
    a = x0
    b = x0 + step
    fa = fx0
    fb = f(b)
    
    # If f(a) and f(b) don't have opposite signs, try other direction or expand
    # Standard approach: expand both ways or just one way if we know the sign.
    # Since we don't know where the root is, symmetric expansion is safer.
    
    # Let's try a few small steps first to catch close roots
    # x0 - step, x0 + step
    
    found_bracket = False
    a_val = None
    b_val = None
    fa_val = None
    fb_val = None
    
    # Initial check with small step
    for i in range(10):
        step = 10.0 ** (i - 5) # 1e-5 to 1e4
        a_candidate = x0 - step
        b_candidate = x0 + step
        
        if time.time() > deadline:
            # Fallback to Newton if time is up
            break
            
        fa_c = f(a_candidate)
        fb_c = f(b_candidate)
        
        if fa_c * fb_c < 0:
            found_bracket = True
            a_val = a_candidate
            b_val = b_candidate
            fa_val = fa_c
            fb_val = fb_c
            break
        
        # Also check if the endpoints themselves are roots
        if abs(fa_c) <= 1e-8:
            return a_candidate
        if abs(fb_c) <= 1e-8:
            return b_candidate

    if found_bracket:
        try:
            # brentq is very efficient
            root = brentq(f, a_val, b_val, xtol=1e-12, rtol=1e-12, maxiter=100)
            return root
        except Exception:
            pass

    # If no bracket found, or brentq failed, fallback to Newton's method
    # This handles cases where the root is isolated or bracketing is hard
    x = x0
    for _ in range(100):
        if time.time() > deadline:
            break
        fx = f(x)
        if abs(fx) <= 1e-8:
            return x
        
        h = 1e-6 * max(1.0, abs(x))
        d = (f(x + h) - fx) / h
        if d == 0:
            break
        x_new = x - fx / d
        
        # Damping if step is too large
        if abs(x_new - x) > 10 * max(1.0, abs(x)):
             x_new = x + 10 * max(1.0, abs(x)) * (1 if x_new > x else -1)
             
        x = x_new

    return x
