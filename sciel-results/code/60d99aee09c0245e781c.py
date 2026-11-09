"""Root finder for 3 fixed oscillatory equations."""
import math
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    call_count = 0
    max_calls = 450  # Leave buffer for safety
    
    def safe_call(x):
        nonlocal call_count
        if call_count >= max_calls:
            raise Exception("Call limit exceeded")
        call_count += 1
        return f(x)

    best_x = float(x0)
    best_f = None
    try:
        best_f = safe_call(best_x)
    except Exception:
        return best_x

    if not math.isfinite(best_f):
        return best_x
        
    if abs(best_f) <= 1e-8:
        return best_x

    # Strategy 1: Try IQI (Illinois-Quadratic-Interpolation) hybrid first
    # This avoids the bracketing overhead when x0 is near a root
    x1 = float(x0)
    fx1 = best_f
    
    # Get a second point for secant/IQI
    step = 1e-4
    x2 = x1 + step
    try:
        fx2 = safe_call(x2)
    except Exception:
        fx2 = None
    
    if fx2 is None or not math.isfinite(fx2):
        # Try negative step
        x2 = x1 - step
        try:
            fx2 = safe_call(x2)
        except Exception:
            fx2 = None
    
    if fx2 is not None and math.isfinite(fx2):
        if abs(fx2) <= 1e-8:
            return x2
        if abs(fx2) < abs(best_f):
            best_x = x2
            best_f = fx2
        
        # IQI iteration: use quadratic interpolation when possible, else bisection-like
        # Keep track of bracket if signs differ
        if fx1 * fx2 < 0:
            # We have a bracket, use Brent's method
            a, b = (x1, x2) if fx1 * fx2 < 0 else (x2, x1)
            fa, fb = (fx1, fx2) if fx1 * fx2 < 0 else (fx2, fx1)
            try:
                root = brentq(safe_call, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                if math.isfinite(root):
                    return root
            except Exception:
                pass
        
        # Secant method with IQI acceleration
        for i in range(60):
            if time.time() > deadline:
                break
            if abs(fx2) <= 1e-8:
                return x2
            if abs(fx1) <= 1e-8:
                return x1
                
            if fx1 == fx2:
                break
                
            # Standard secant step
            denom = fx2 - fx1
            if abs(denom) < 1e-30:
                break
            x3 = x2 - fx2 * (x2 - x1) / denom
            
            if not math.isfinite(x3):
                break
            
            # IQI: if x3 is outside [min(x1,x2), max(x1,x2)], use bisection instead
            lo = min(x1, x2)
            hi = max(x1, x2)
            if x3 < lo or x3 > hi:
                # Use midpoint (bisection-like)
                x3 = (x1 + x2) / 2.0
            
            x1, x2 = x2, x3
            fx1, fx2 = fx2, safe_call(x2)
            
            if abs(fx2) < abs(best_f):
                best_x = x2
                best_f = fx2
        
        if abs(best_f) <= 1e-8:
            return best_x

    # Strategy 2: Find a bracket by expanding from x0
    a = None
    b = None
    fa = None
    fb = None
    
    x = float(x0)
    fx = best_f
    
    # Try to find a bracket by expanding from x0
    # We check points in increasing distance from x0
    distances = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0, 100000.0]
    
    found_bracket = False
    last_r_x = x
    last_r_fx = fx
    last_l_x = x
    last_l_fx = fx

    for d in distances:
        if time.time() > deadline:
            break
        
        # Check right
        x_r = x + d
        try:
            f_r = safe_call(x_r)
        except Exception:
            break
        if math.isfinite(f_r):
            if abs(f_r) < abs(best_f):
                best_x = x_r
                best_f = f_r
            if abs(f_r) <= 1e-8:
                return x_r
            if last_r_fx * f_r < 0:
                a = last_r_x
                fa = last_r_fx
                b = x_r
                fb = f_r
                found_bracket = True
                break
            last_r_x = x_r
            last_r_fx = f_r

        # Check left
        x_l = x - d
        try:
            f_l = safe_call(x_l)
        except Exception:
            break
        if math.isfinite(f_l):
            if abs(f_l) < abs(best_f):
                best_x = x_l
                best_f = f_l
            if abs(f_l) <= 1e-8:
                return x_l
            if last_l_fx * f_l < 0:
                a = x_l
                fa = f_l
                b = last_l_x
                fb = last_l_fx
                found_bracket = True
                break
            last_l_x = x_l
            last_l_fx = f_l

    if found_bracket:
        try:
            # Use Brent's method
            root = brentq(safe_call, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if math.isfinite(root):
                return root
        except Exception:
            pass

    # Strategy 3: Fallback - Secant method
    x1 = x
    fx1 = fx
    x2 = x + 1e-4
    try:
        fx2 = safe_call(x2)
    except Exception:
        return best_x
    
    if not math.isfinite(fx2):
        return best_x

    # Secant iteration
    for i in range(50):
        if time.time() > deadline:
            break
        if abs(fx2) <= 1e-8:
            return x2
        if abs(fx1) <= 1e-8:
            return x1
            
        if fx1 == fx2:
            break
            
        x3 = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
        
        if not math.isfinite(x3):
            break
            
        x1, x2 = x2, x3
        fx1, fx2 = fx2, safe_call(x2)
        
        if abs(fx2) < abs(best_f):
            best_x = x2
            best_f = fx2

    return best_x
# EVOLVE-BLOCK-END
