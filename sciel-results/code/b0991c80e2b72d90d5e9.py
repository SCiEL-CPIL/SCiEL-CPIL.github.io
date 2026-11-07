import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    x = float(x0)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # Strategy 1: Secant Method
    # Initial points
    x0_sec = x
    f0_sec = fx
    
    # Choose initial step
    # Scale step by magnitude of x to handle large/small x
    delta = 1e-4 * max(1.0, abs(x0_sec))
    x1_sec = x0_sec + delta
    f1_sec = f(x1_sec)
    
    if abs(f1_sec) <= 1e-8:
        return x1_sec
    
    # Iterate Secant
    for i in range(50):
        if f1_sec == f0_sec:
            break # Avoid division by zero, fallback
        
        # Secant formula
        x2_sec = x1_sec - f1_sec * (x1_sec - x0_sec) / (f1_sec - f0_sec)
        
        # Check for non-finite
        if not np.isfinite(x2_sec):
            break
            
        f2_sec = f(x2_sec)
        
        if abs(f2_sec) <= 1e-8:
            return float(x2_sec)
            
        # Check for convergence
        if abs(x2_sec - x1_sec) < 1e-12:
            # If x converged but f is not small enough, it might be a flat region
            # Return best so far
            if abs(f2_sec) < abs(f1_sec):
                return float(x2_sec)
            else:
                return float(x1_sec)
                
        # Check for divergence
        # If the step size is growing too fast, or x is going to infinity
        if abs(x2_sec - x1_sec) > 10.0 * abs(x1_sec - x0_sec) + 1e-6:
            break
        if abs(x2_sec) > 1e10:
            break
            
        # Update points
        x0_sec = x1_sec
        f0_sec = f1_sec
        x1_sec = x2_sec
        f1_sec = f2_sec
        
    # Strategy 2: Fallback to Bracketing (Brent)
    # Use a wider initial bracket
    a = x
    fa = fx
    h = 1e-3 * max(1.0, abs(x))
    
    # Expand bracket
    for i in range(50):
        # Try positive direction
        x_pos = x + h
        fx_pos = f(x_pos)
        if abs(fx_pos) <= 1e-8:
            return x_pos
        if fa * fx_pos < 0:
            try:
                root = brentq(f, x, x_pos, xtol=1e-12, rtol=1e-12, maxiter=100)
                return float(root)
            except Exception:
                pass
        
        # Try negative direction
        x_neg = x - h
        fx_neg = f(x_neg)
        if abs(fx_neg) <= 1e-8:
            return x_neg
        if fa * fx_neg < 0:
            try:
                root = brentq(f, x_neg, x, xtol=1e-12, rtol=1e-12, maxiter=100)
                return float(root)
            except Exception:
                pass
        
        h *= 2.0 # Expand faster
        if h > 1e6:
            break
            
    # Strategy 3: Final Fallback - Return best guess
    # We don't have a global best, so return x0 or last secant point
    # If Secant ran, x1_sec is likely closer than x0
    if 'x1_sec' in locals() and np.isfinite(x1_sec):
         # Compare f(x1_sec) and f(x0)
         # We have f1_sec
         if abs(f1_sec) <= abs(fx):
             return float(x1_sec)
             
    return x
