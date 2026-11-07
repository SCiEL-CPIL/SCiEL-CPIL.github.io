import time
import math

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    Strategy:
      1. Quick Newton phase (limited steps, low cost per step).
      2. If Newton succeeds, return.
      3. Otherwise, use a robust bracketing method (bisection-like with adaptive expansion)
         to find a sign change, then apply Brent's method.
    Deadline guard: return best candidate if approaching time limit.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin within 120s limit
    
    x = float(x0)
    best_x = x
    best_fx = f(x)
    best_abs_fx = abs(best_fx)
    
    if best_abs_fx <= 1e-8:
        return x
    
    # Phase 1: Newton's method with finite-difference derivative (forward difference)
    newton_steps = 0
    for i in range(30):
        if time.time() > deadline:
            return best_x
        
        fx = f(x)
        if not math.isfinite(fx):
            break
        abs_fx = abs(fx)
        if abs_fx <= 1e-8:
            return x
        if abs_fx < best_abs_fx:
            best_x = x
            best_fx = fx
            best_abs_fx = abs_fx
        
        # Forward difference for derivative
        h = 1e-7 * max(1.0, abs(x))
        fx_plus = f(x + h)
        if not math.isfinite(fx_plus):
            break
        d = (fx_plus - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        
        # Damping: if step is too large, damp it
        if abs(x_new - x) > 10 * max(1.0, abs(x)):
            x_new = x + 0.5 * (x_new - x)
        
        x = x_new
        newton_steps += 1
    
    # Check if Newton converged
    fx = f(x)
    if math.isfinite(fx) and abs(fx) <= 1e-8:
        return x
    if math.isfinite(fx) and abs(fx) < best_abs_fx:
        best_x = x
        best_fx = fx
        best_abs_fx = abs(fx)
    
    # Phase 2: Bracketing + Bisection/Brent
    # Try to find a bracket [a, b] where f(a) and f(b) have opposite signs
    a = x
    fa = f(a)
    if not math.isfinite(fa):
        a = x0
        fa = f(a)
        if not math.isfinite(fa):
            return best_x
    
    # Expand bracket from a
    step = max(1.0, abs(a)) * 0.1
    b = a
    fb = fa
    found_bracket = False
    
    for direction in [1, -1]:
        if time.time() > deadline:
            return best_x
        
        b = a + direction * step
        fb = f(b)
        if not math.isfinite(fb):
            continue
        
        # Check if we have a sign change
        if fa * fb <= 0:
            found_bracket = True
            break
        
        # Expand
        step *= 2
        if step > 1e10:
            break
    
    if not found_bracket:
        # Try a different strategy: scan around x0
        scan_range = max(10.0, abs(x0) * 2)
        n_scan = 50
        for i in range(n_scan):
            if time.time() > deadline:
                return best_x
            
            xi = x0 - scan_range + 2 * scan_range * i / (n_scan - 1)
            fxi = f(xi)
            if not math.isfinite(fxi):
                continue
            if abs(fxi) < best_abs_fx:
                best_x = xi
                best_fx = fxi
                best_abs_fx = abs(fxi)
            if abs(fxi) <= 1e-8:
                return xi
            
            # Check sign change with previous
            if i > 0:
                prev_x = x0 - scan_range + 2 * scan_range * (i - 1) / (n_scan - 1)
                prev_f = f(prev_x) if i < n_scan else None
                # We didn't store previous, so just check consecutive
                if math.isfinite(fxi) and math.isfinite(f(prev_x)):
                    if fxi * f(prev_x) <= 0:
                        a = prev_x
                        b = xi
                        fa = f(prev_x)
                        fb = fxi
                        found_bracket = True
                        break
    
    if found_bracket:
        # Use bisection to narrow down, then polish
        lo = min(a, b)
        hi = max(a, b)
        flo = f(lo)
        fhi = f(hi)
        
        # Ensure flo and fhi have opposite signs
        if flo * fhi > 0:
            return best_x
        
        for i in range(100):
            if time.time() > deadline:
                return best_x
            
            mid = (lo + hi) / 2
            fmid = f(mid)
            if not math.isfinite(fmid):
                return best_x
            
            if abs(fmid) <= 1e-8:
                return mid
            
            if abs(fmid) < best_abs_fx:
                best_x = mid
                best_fx = fmid
                best_abs_fx = abs(fmid)
            
            if flo * fmid <= 0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid
            
            # Check convergence
            if abs(hi - lo) < 1e-12 * max(1.0, abs(mid)):
                break
        
        # Final check
        mid = (lo + hi) / 2
        fmid = f(mid)
        if math.isfinite(fmid) and abs(fmid) <= 1e-8:
            return mid
        if abs(fmid) < best_abs_fx:
            best_x = mid
            best_abs_fx = abs(fmid)
    
    return best_x
