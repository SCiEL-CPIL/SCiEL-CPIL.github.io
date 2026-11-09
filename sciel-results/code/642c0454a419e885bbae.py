import math
import time
# EVOLVE-BLOCK-START
def find_root(f, x0):
    deadline = time.time() + 145.0
    tol = 1e-12  # Tighter internal tolerance
    max_calls = 450  # Safety margin for 500 limit
    
    x0 = float(x0)
    calls = [0]

    def fc(x):
        calls[0] += 1
        try:
            val = f(x)
        except Exception:
            return float('nan')
        if val is None or not math.isfinite(val):
            return float('nan')
        return float(val)

    # State
    best_x = x0
    best_f = fc(x0)
    best_abs = abs(best_f) if math.isfinite(best_f) else float('inf')
    
    if best_abs <= tol:
        return best_x

    # Function to update best
    def update_best(x, fx):
        nonlocal best_x, best_f, best_abs
        if not math.isfinite(x) or not math.isfinite(fx):
            return
        afx = abs(fx)
        if afx < best_abs:
            best_x = x
            best_f = fx
            best_abs = afx
        if afx <= tol:
            return True
        return False

    # Brent's Method implementation
    # a, b are the bracket endpoints where f(a)*f(b) < 0
    def brent(a, b, fa, fb):
        nonlocal best_x, best_f, best_abs
        c = a
        fc_val = fa
        d = b - a
        e = d
        A = 0.0
        B = 0.0
        
        for _ in range(100):
            if time.time() > deadline or calls[0] >= max_calls:
                break
            if fb * fc_val > 0:
                c = a
                fc_val = fa
                d = b - a
                e = d
            if abs(fc_val) < abs(fb):
                a = b
                b = c
                c = a
                fa = fb
                fb = fc_val
                fc_val = fa
            tol1 = 2.2e-16 * abs(b) + 5e-324
            m = 0.5 * (c - b)
            if abs(m) <= tol1 or fb == 0.0:
                return b
            
            if abs(e) >= tol1 and abs(A) >= abs(B):
                s = B / A
                e = d
                d = b - s
            else:
                d = m
                e = d
                A = fc_val - fb
                B = (fc_val - fa) * m / (fa - fb) if fa != fb else 0.0
                A = (fb - fc_val) / (b - c) if b != c else 0.0
                # Correcting A and B calculation for Inverse Quadratic Interpolation
                # Standard Brent's method logic:
                # A = (fb - fc) / (b - c)
                # B = (fa - fc) / (a - c)
                # If |A| >= |B|, use secant from (b, fb) and (c, fc)
                # Else use quadratic interpolation
                pass # We will implement a simpler robust version below to avoid bugs in this snippet logic
                
        # Fallback to bisection if Brent loop exits without convergence
        return b

    # Simpler, more robust "Hybrid Secant/Bisection" 
    # 1. Try to find bracket
    # 2. Use Secant if no bracket, Bisection if bracket
    
    # Phase 1: Find a bracket
    a = x0
    fa = best_f
    b = None
    fb = None
    
    # Initial step
    step = max(1e-6, abs(x0) * 1e-6)
    # Try expanding to find sign change
    for i in range(50):
        if time.time() > deadline or calls[0] >= max_calls:
            break
        # Try positive direction
        b_try = a + step
        fb_try = fc(b_try)
        if math.isfinite(fb_try):
            if update_best(b_try, fb_try):
                return best_x
            if fa * fb_try < 0:
                b = b_try
                fb = fb_try
                break
            # If no sign change, try negative direction if we haven't bracketed yet
            # But expanding in one direction is usually better for oscillatory if we pick the right side
            # Let's just expand step
            step *= 2.0
            if abs(b_try) > 1e15:
                break
        else:
            break
            
    if b is None:
        # Try negative direction
        step = max(1e-6, abs(x0) * 1e-6)
        for i in range(50):
            if time.time() > deadline or calls[0] >= max_calls:
                break
            b_try = a - step
            fb_try = fc(b_try)
            if math.isfinite(fb_try):
                if update_best(b_try, fb_try):
                    return best_x
                if fa * fb_try < 0:
                    b = b_try
                    fb = fb_try
                    break
                step *= 2.0
                if abs(b_try) > 1e15:
                    break
            else:
                break

    if b is not None:
        # Phase 2: Refine using Hybrid Secant/Bisection (Brent-like)
        # We have [a, b] with sign change
        # Use Bisection but accelerate with Secant if safe
        for _ in range(100):
            if time.time() > deadline or calls[0] >= max_calls:
                break
            if best_abs <= tol:
                break
                
            mid = (a + b) / 2.0
            fmid = fc(mid)
            if not math.isfinite(fmid):
                # If mid fails, shrink bracket? Or just bisect further?
                # If fmid is nan, we can't use it. Just bisect.
                # Actually if f is undefined at mid, we might be in a bad region.
                # Let's just continue bisection.
                if fa * fmid < 0:
                    b = mid
                    fb = fmid
                else:
                    a = mid
                    fa = fmid
                continue
            
            update_best(mid, fmid)
            if best_abs <= tol:
                break
                
            if fa * fmid < 0:
                b = mid
                fb = fmid
            else:
                a = mid
                fa = fmid
                
            # Acceleration: Try Secant step from (a, fa) and (b, fb)
            # Only if the step stays within [a, b]
            if abs(fb - fa) > 1e-15:
                x_sec = b - fb * (b - a) / (fb - fa)
                if a < x_sec < b:
                    f_sec = fc(x_sec)
                    if math.isfinite(f_sec):
                        update_best(x_sec, f_sec)
                        if best_abs <= tol:
                            break
                        # Update bracket based on secant result
                        if fa * f_sec < 0:
                            b = x_sec
                            fb = f_sec
                        elif fb * f_sec < 0:
                            a = x_sec
                            fa = f_sec
                        # If no sign change with secant, keep current bracket [a,b]
                        # and continue bisection
            # Convergence check
            if abs(b - a) < 1e-14 * max(1.0, abs(a)):
                break
    else:
        # No bracket found. Try Secant/Newton from x0
        # Use a few secant steps
        x1 = x0 + max(1e-6, abs(x0) * 1e-6)
        f1 = fc(x1)
        if math.isfinite(f1):
            update_best(x1, f1)
            xa, fa_xa = x0, fa
            xb, fb_xb = x1, f1
            for _ in range(50):
                if time.time() > deadline or calls[0] >= max_calls:
                    break
                if best_abs <= tol:
                    break
                if abs(fb_xb - fa_xa) < 1e-15:
                    break
                xc = xb - fb_xb * (xb - xa) / (fb_xb - fa_xa)
                if not math.isfinite(xc):
                    break
                fc_xc = fc(xc)
                if not math.isfinite(fc_xc):
                    break
                update_best(xc, fc_xc)
                if best_abs <= tol:
                    break
                # Shift window
                xa, fa_xa = xb, fb_xb
                xb, fb_xb = xc, fc_xc
                if abs(xc - xb) < 1e-14 * max(1.0, abs(xc)):
                    break

    return best_x
# EVOLVE-BLOCK-END
