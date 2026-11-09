"""Root finder for 3 fixed oscillatory equations."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin well under 150s

    x = float(x0)
    
    # Check initial point
    try:
        fx = f(x)
    except Exception:
        return x
    if not math.isfinite(fx):
        return x
    if abs(fx) <= 1e-8:
        return x

    x_best = x
    f_best = fx

    def track(x, fx):
        nonlocal x_best, f_best
        if math.isfinite(fx) and abs(fx) < abs(f_best):
            x_best = x
            f_best = fx

    track(x, fx)

    # 1. Global Adaptive Scan to find a bracket
    # We scan outwards from x0 on a logarithmic scale.
    # This is more robust than geometric expansion from a single side 
    # and helps find sign changes in oscillatory functions.
    
    a, b = None, None
    fa, fb = None, None
    
    # We keep track of the last evaluated point to check for sign change
    # against the current point.
    # We start with a small step and double it.
    
    # Initial points: x0 and a small step away.
    # To avoid missing roots very close to x0, we check a tiny step first.
    
    points = []
    # Add x0
    points.append((x, fx))
    
    step = 1e-4
    # Scan up to 1e6 or until deadline
    # We scan in both directions: x0 - step, x0 + step, x0 - 2*step, x0 + 2*step, etc.
    # Actually, a simpler robust scan is:
    # d = 1e-3, 1e-2, 0.1, 1, 10, 100, 1000...
    # Check sign change between (x0, x0+d), (x0+d, x0+2d), etc?
    # No, just check pairs of consecutive evaluated points.
    
    # Let's use a specific list of scales to cover a wide range efficiently.
    scales = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    
    # We will generate candidates: x0 + s, x0 - s
    # Sort them by distance from x0? 
    # Better: Interleave small and large? 
    # For oscillatory functions, the period might be anything. 
    # A logarithmic scan is best.
    
    candidates = []
    for s in scales:
        candidates.append(x + s)
        candidates.append(x - s)
    
    # Remove duplicates and x0 itself if present (though x0 is already in points)
    # Evaluate in order of increasing distance from x0 to find local roots first?
    # Or just evaluate all?
    # Evaluating all 22 points is cheap (< 22 calls).
    # Let's evaluate them.
    
    # To ensure we find the *closest* bracket, we should probably check closer points first.
    # Let's sort candidates by absolute distance from x0.
    candidates.sort(key=lambda c: abs(c - x))
    
    prev_x, prev_f = x, fx
    
    for c in candidates:
        if time.time() > deadline:
            break
        if c == x:
            continue
        try:
            fc = f(c)
        except Exception:
            continue
            
        if not math.isfinite(fc):
            continue
            
        track(c, fc)
        
        # Check sign change with previous point
        if prev_f * fc < 0:
            a, b = min(prev_x, c), max(prev_x, c)
            fa, fb = prev_f if prev_x == a else fc, prev_f if prev_x == b else fc
            # Ensure fa corresponds to a, fb to b
            if a == prev_x:
                fa, fb = prev_f, fc
            else:
                fa, fb = fc, prev_f
            break
            
        prev_x, prev_f = c, fc
        
    # If no bracket found in the fixed scales, try a wider geometric expansion
    if a is None:
        step = 10000.0
        prev_x, prev_f = x, fx
        # Continue from where we left off? 
        # The last candidate was the furthest one.
        # Let's just do a quick expansion from the furthest point found or x0.
        # Actually, if the scales didn't work, the root is likely very far or the function is pathological.
        # Let's try a few more large steps.
        for i in range(10):
            if time.time() > deadline:
                break
            c = x + step
            try:
                fc = f(c)
            except Exception:
                break
            if not math.isfinite(fc):
                step *= 2
                continue
            if prev_f * fc < 0:
                a, b = min(prev_x, c), max(prev_x, c)
                if a == prev_x:
                    fa, fb = prev_f, fc
                else:
                    fa, fb = fc, prev_f
                break
            prev_x, prev_f = c, fc
            step *= 2
            
            c = x - step
            try:
                fc = f(c)
            except Exception:
                break
            if not math.isfinite(fc):
                continue
            if prev_f * fc < 0:
                a, b = min(prev_x, c), max(prev_x, c)
                if a == prev_x:
                    fa, fb = prev_f, fc
                else:
                    fa, fb = fc, prev_f
                break
            prev_x, prev_f = c, fc

    # If still no bracket, use Newton/Secant from best guess
    if a is None:
        # Try a few Newton steps from x_best
        x_curr = x_best
        f_curr = f_best
        for i in range(20):
            if time.time() > deadline:
                break
            h = 1e-7 * max(1.0, abs(x_curr))
            try:
                f_curr_h = f(x_curr + h)
            except Exception:
                break
            if not math.isfinite(f_curr_h):
                break
            d = (f_curr_h - f_curr) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x_curr - f_curr / d
            if not math.isfinite(x_new):
                break
            x_curr = x_new
            try:
                f_curr = f(x_curr)
            except Exception:
                break
            if not math.isfinite(f_curr):
                break
            track(x_curr, f_curr)
            if abs(f_curr) <= 1e-8:
                return x_curr
        return x_best

    # 2. Refine using IQI/Secant Hybrid
    # We have bracket [a, b] with fa, fb
    # Ensure |fa| <= |fb| for standard Illinois logic, but we will use a more general safeguarded method.
    
    # Points for interpolation:
    # x0, f0 (current best guess inside or endpoint)
    # x1, f1
    # x2, f2
    
    # Let's maintain a list of the last 3 points used.
    # Initialize with a, b and a midpoint? Or just a, b?
    # Secant needs 2 points. IQI needs 3.
    # Let's start with Secant from a and b.
    
    x1, f1 = a, fa
    x2, f2 = b, fb
    
    # Ensure we don't divide by zero or have identical points
    if x1 == x2:
        return x1 if abs(f1) < abs(f2) else x2
        
    for i in range(100):
        if time.time() > deadline:
            break
            
        if abs(f1) <= 1e-8:
            return x1
        if abs(f2) <= 1e-8:
            return x2
            
        # Try Secant Step
        # x_sec = x2 - f2 * (x2 - x1) / (f2 - f1)
        if f1 != f2:
            x_sec = x2 - f2 * (x2 - x1) / (f2 - f1)
        else:
            x_sec = 0.5 * (x1 + x2) # Fallback to bisection if slope is 0
            
        # Safeguard: If x_sec is outside [min(x1,x2), max(x1,x2)] or too close to endpoints, use Bisection
        # Actually, for root finding, we want to stay within the bracket.
        # If x_sec is not in (min(x1,x2), max(x1,x2)), use Bisection.
        
        if x_sec <= min(x1, x2) or x_sec >= max(x1, x2):
            x_new = 0.5 * (x1 + x2)
        else:
            x_new = x_sec
            
        try:
            f_new = f(x_new)
        except Exception:
            # If f fails, return best found so far
            break
            
        if not math.isfinite(f_new):
            break
            
        if abs(f_new) <= 1e-8:
            return x_new
            
        track(x_new, f_new)
        
        # Update bracket
        # We need to know which side x_new is on and update the corresponding endpoint.
        # If f1 * f_new < 0, root is in [x1, x_new]
        # Else root is in [x_new, x2]
        
        if f1 * f_new < 0:
            # New bracket [x1, x_new]
            # For Secant/IQI, we want to keep the points that define the curve.
            # We discard x2 and use x_new as the new x2? 
            # Or do we keep x1, x2, x_new for IQI?
            # Simple Secant: update one of the points.
            # To maintain superlinear convergence, we should replace the point with the larger |f|.
            # But for safety, let's just shift the window:
            # If we used Secant, we have 3 points: x1, x2, x_new.
            # The new bracket is [x1, x_new] or [x_new, x2].
            # Let's just use the two endpoints of the new bracket for the next Secant step.
            # This is effectively a "Safeguarded Secant".
            
            if abs(f1) > abs(f_new):
                # Replace x1 with x_new? No, x1 is the other end of the bracket.
                # The bracket is [x1, x_new].
                # We should probably use x_new and x1.
                # But wait, if we just use the endpoints, we lose the "curvature" info of the 3rd point.
                # However, Safeguarded Secant (using only endpoints of current bracket) is robust and fast.
                # Let's stick to using the current bracket endpoints for the next Secant step.
                x2, f2 = x_new, f_new
            else:
                # f1 is smaller, keep x1. Replace x2 with x_new.
                x2, f2 = x_new, f_new
                # Wait, if f1 * f_new < 0, the root is between x1 and x_new.
                # So the new bracket is [x1, x_new].
                # We should set x2 = x_new.
                x2, f2 = x_new, f_new
        else:
            # Root is in [x_new, x2]
            x1, f1 = x_new, f_new
            
        # Convergence check
        if abs(x2 - x1) < 1e-12 * max(1.0, abs(x1), abs(x2)):
            break

    # Return best
    if abs(f1) < abs(f2):
        return x1
    else:
        return x2

# EVOLVE-BLOCK-END
