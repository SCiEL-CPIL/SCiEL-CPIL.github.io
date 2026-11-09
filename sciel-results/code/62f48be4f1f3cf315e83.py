"""Root finder for 3 fixed oscillatory equations."""
import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    deadline = time.time() + 115.0
    call_count = 0
    max_calls = 480  # Leave buffer for safety
    
    def safe_call(x):
        nonlocal call_count
        if call_count >= max_calls:
            raise Exception("Call limit exceeded")
        call_count += 1
        return f(x)
    
    x0 = float(x0)
    best_x = x0
    best_f = None
    
    try:
        best_f = safe_call(x0)
    except Exception:
        return x0
    
    if not math.isfinite(best_f):
        return x0
    
    if abs(best_f) <= 1e-8:
        return x0
    
    # Strategy: Try to find a bracket using adaptive step sizes
    # For oscillatory functions, roots are often at predictable intervals
    
    # First, try small steps to see if we're near a root
    small_steps = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
    
    fx0 = best_f
    x = x0
    
    # Try to find a sign change with small steps
    for step in small_steps:
        if time.time() > deadline:
            break
        if call_count >= max_calls - 50:
            break
            
        # Try positive direction
        x_pos = x + step
        try:
            f_pos = safe_call(x_pos)
        except Exception:
            f_pos = None
        
        if f_pos is not None and math.isfinite(f_pos):
            if abs(f_pos) <= 1e-8:
                return x_pos
            if abs(f_pos) < abs(best_f):
                best_x = x_pos
                best_f = f_pos
            if fx0 * f_pos < 0:
                # Found bracket [x, x_pos]
                a, b = (x, x_pos) if fx0 < f_pos else (x_pos, x)
                fa, fb = (fx0, f_pos) if fx0 < f_pos else (f_pos, fx0)
                try:
                    root = brentq(safe_call, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                    if math.isfinite(root):
                        return root
                except Exception:
                    pass
        
        # Try negative direction
        x_neg = x - step
        try:
            f_neg = safe_call(x_neg)
        except Exception:
            f_neg = None
        
        if f_neg is not None and math.isfinite(f_neg):
            if abs(f_neg) <= 1e-8:
                return x_neg
            if abs(f_neg) < abs(best_f):
                best_x = x_neg
                best_f = f_neg
            if fx0 * f_neg < 0:
                # Found bracket [x_neg, x]
                a, b = (x_neg, x) if f_neg < fx0 else (x, x_neg)
                fa, fb = (f_neg, fx0) if f_neg < fx0 else (fx0, f_neg)
                try:
                    root = brentq(safe_call, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                    if math.isfinite(root):
                        return root
                except Exception:
                    pass
    
    # If no bracket found with small steps, use secant method with adaptive stepping
    # For oscillatory functions, we need to be careful about which root we converge to
    
    # Use secant method with two initial points
    x1 = x0
    f1 = best_f
    
    # Choose second point based on the function's local behavior
    # Try a moderate step
    step2 = 0.01
    x2 = x1 + step2
    try:
        f2 = safe_call(x2)
    except Exception:
        x2 = x1 - step2
        try:
            f2 = safe_call(x2)
        except Exception:
            return best_x
    
    if not math.isfinite(f2):
        return best_x
    
    if abs(f2) <= 1e-8:
        return x2
    
    if abs(f2) < abs(best_f):
        best_x = x2
        best_f = f2
    
    # Secant iteration with safeguards
    for i in range(80):
        if time.time() > deadline:
            break
        if call_count >= max_calls - 10:
            break
        
        if abs(f2) <= 1e-8:
            return x2
        if abs(f1) <= 1e-8:
            return x1
        
        if f1 == f2:
            break
        
        denom = f2 - f1
        if abs(denom) < 1e-30:
            break
        
        x3 = x2 - f2 * (x2 - x1) / denom
        
        if not math.isfinite(x3):
            break
        
        # Safeguard: if x3 is too far from both x1 and x2, use bisection
        mid = (x1 + x2) / 2.0
        span = abs(x2 - x1)
        if abs(x3 - mid) > 2.0 * span:
            # Use a point closer to the midpoint
            x3 = x2 - f2 * (x2 - x1) / (f2 - f1) * 0.5
            if not math.isfinite(x3):
                x3 = mid
        
        x1, x2 = x2, x3
        f1, f2 = f2, safe_call(x2)
        
        if abs(f2) < abs(best_f):
            best_x = x2
            best_f = f2
    
    if abs(best_f) <= 1e-8:
        return best_x
    
    # Final fallback: try to find a bracket with larger steps
    # Use exponential expansion
    a = x0
    fa = best_f
    b = None
    fb = None
    
    for mult in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]:
        if time.time() > deadline:
            break
        if call_count >= max_calls - 5:
            break
        
        # Try positive
        b_try = x0 + mult
        try:
            fb_try = safe_call(b_try)
        except Exception:
            fb_try = None
        
        if fb_try is not None and math.isfinite(fb_try):
            if abs(fb_try) <= 1e-8:
                return b_try
            if abs(fb_try) < abs(best_f):
                best_x = b_try
                best_f = fb_try
            if fa * fb_try < 0:
                b = b_try
                fb = fb_try
                break
        
        # Try negative
        b_try = x0 - mult
        try:
            fb_try = safe_call(b_try)
        except Exception:
            fb_try = None
        
        if fb_try is not None and math.isfinite(fb_try):
            if abs(fb_try) <= 1e-8:
                return b_try
            if abs(fb_try) < abs(best_f):
                best_x = b_try
                best_f = fb_try
            if fa * fb_try < 0:
                b = b_try
                fb = fb_try
                break
    
    if b is not None and fb is not None:
        try:
            root = brentq(safe_call, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if math.isfinite(root):
                return root
        except Exception:
            pass
    
    return best_x
