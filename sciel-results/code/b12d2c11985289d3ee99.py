"""Root finder for 3 fixed oscillatory equations with optimized call budget."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x with |f(x)| <= 1e-8.
    
    Strategy:
    1. Adaptive bracketing search: dense near x0, then geometric expansion.
    2. Illinois-style hybrid secant/bisection for fast bracketed convergence.
    3. Fallback: multiple secant attempts if no bracket found.
    4. Final polish pass for tight tolerance.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    tol = 1e-8
    
    x0 = float(x0)
    
    # 1. Initial Evaluation
    try:
        f0 = f(x0)
    except Exception:
        x_try = x0 + 1e-6
        try:
            f0 = f(x_try)
            x0 = x_try
        except Exception:
            return x0
    
    if not math.isfinite(f0):
        return x0
        
    if abs(f0) <= tol:
        return x0

    # Track best point found
    best_x = x0
    best_f = f0
    best_abs = abs(f0)
    
    def update_best(x, fx):
        nonlocal best_x, best_f, best_abs
        if math.isfinite(x) and math.isfinite(fx):
            af = abs(fx)
            if af < best_abs:
                best_x = x
                best_f = fx
                best_abs = af

    def time_ok():
        return time.time() < deadline

    # 2. Adaptive Bracketing Search
    # Phase 1: Dense search near x0 with small steps
    # Phase 2: Geometric expansion
    a = None
    b = None
    fa = None
    fb = None
    
    # Dense near x0: steps of 1e-5, 1e-4, 1e-3, 1e-2, 1e-1
    dense_deltas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    
    for d in dense_deltas:
        if not time_ok():
            break
            
        # Try left
        x_left = x0 - d
        try:
            fl = f(x_left)
        except Exception:
            fl = float('nan')
            
        if math.isfinite(fl):
            update_best(x_left, fl)
            if abs(fl) <= tol:
                return x_left
            if fa is None:
                fa = fl
                a = x_left
            if fa * f0 < 0:
                a, fa = x_left, fl
                b, fb = x0, f0
                break
        
        # Try right
        x_right = x0 + d
        try:
            fr = f(x_right)
        except Exception:
            fr = float('nan')
            
        if math.isfinite(fr):
            update_best(x_right, fr)
            if abs(fr) <= tol:
                return x_right
            if fb is None:
                fb = fr
                b = x_right
            if f0 * fr < 0:
                a, fa = x0, f0
                b, fb = x_right, fr
                break

    # If bracket found, use Illinois hybrid method
    if a is not None and b is not None:
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        # Illinois algorithm: faster than bisection, maintains bracket
        for _ in range(100):
            if not time_ok():
                break
                
            if abs(fa) <= tol:
                return a
            if abs(fb) <= tol:
                return b
            
            # Secant estimate
            denom = fb - fa
            if abs(denom) < 1e-300 or not math.isfinite(denom):
                x_new = (a + b) / 2
            else:
                x_new = b - fb * (b - a) / denom
            
            if not math.isfinite(x_new):
                x_new = (a + b) / 2
            
            # If secant point is outside bracket, use bisection
            if x_new <= a or x_new >= b:
                x_new = (a + b) / 2
            
            try:
                f_new = f(x_new)
            except Exception:
                break
                
            if not math.isfinite(f_new):
                break
                
            update_best(x_new, f_new)
            
            if abs(f_new) <= tol:
                return x_new
            
            # Update bracket
            if fa * f_new < 0:
                # Root in [a, x_new]
                b, fb = x_new, f_new
                fa *= 0.5  # Illinois: dampen the discarded side
            else:
                # Root in [x_new, b]
                a, fa = x_new, f_new
                fb *= 0.5  # Illinois: dampen the discarded side
            
            # Convergence check
            if abs(b - a) < 1e-16 * max(1.0, abs(b)):
                break

        # Return best candidate
        if best_abs <= tol:
            return best_x
        return (a + b) / 2

    else:
        # No bracket found. Try multiple secant strategies.
        # Collect all evaluated points
        points = [(x0, f0)]
        
        # Re-evaluate some strategic points for secant
        strategic = [x0 + 1e-3, x0 - 1e-3, x0 + 1e-2, x0 - 1e-2, 
                     x0 + 0.1, x0 - 0.1, x0 + 1.0, x0 - 1.0,
                     x0 + 5.0, x0 - 5.0, x0 + 20.0, x0 - 20.0]
        
        for sx in strategic:
            if not time_ok():
                break
            try:
                sf = f(sx)
            except Exception:
                continue
            if math.isfinite(sf):
                update_best(sx, sf)
                if abs(sf) <= tol:
                    return sx
                points.append((sx, sf))
        
        # Try secant from multiple pairs
        n = len(points)
        if n >= 2:
            for i in range(min(n, 6)):
                for j in range(i + 1, min(n, i + 4)):
                    if not time_ok():
                        break
                    x1, f1 = points[i]
                    x2, f2 = points[j]
                    
                    if abs(f1) <= tol:
                        return x1
                    if abs(f2) <= tol:
                        return x2
                    
                    denom = f2 - f1
                    if abs(denom) < 1e-300 or not math.isfinite(denom):
                        continue
                    
                    # Run secant for a few iterations
                    xa, fa_ = x1, f1
                    xb, fb_ = x2, f2
                    
                    for _ in range(40):
                        if not time_ok():
                            break
                        if abs(fa_) <= tol:
                            return xa
                        if abs(fb_) <= tol:
                            return xb
                        
                        denom2 = fb_ - fa_
                        if abs(denom2) < 1e-300 or not math.isfinite(denom2):
                            break
                        
                        x_new = xb - fb_ * (xb - xa) / denom2
                        
                        if not math.isfinite(x_new):
                            break
                        
                        try:
                            f_new = f(x_new)
                        except Exception:
                            break
                            
                        if not math.isfinite(f_new):
                            break
                            
                        update_best(x_new, f_new)
                        
                        if abs(f_new) <= tol:
                            return x_new
                        
                        # Simple secant update (no bracket guarantee)
                        xa, fa_ = xb, fb_
                        xb, fb_ = x_new, f_new
                    
                    if best_abs <= tol:
                        return best_x
        
        return best_x

# EVOLVE-BLOCK-END
