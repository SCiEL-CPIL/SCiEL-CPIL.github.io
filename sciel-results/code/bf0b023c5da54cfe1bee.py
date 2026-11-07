import math
import time
from scipy.optimize import brentq

_DEADLINE = time.time() + 140.0
_CALL_BUDGET = 480

def _time_left():
    return _DEADLINE - time.time()

def find_root(f, x0):
    """Find x such that |f(x)| <= 1e-8 for an oscillatory function f."""
    counter = [0]
    max_calls = _CALL_BUDGET

    def wf(x):
        counter[0] += 1
        if counter[0] > max_calls:
            raise RuntimeError("call budget exceeded")
        return f(x)

    best_x = float(x0)
    best_abs = float('inf')

    def try_update(x, val):
        nonlocal best_x, best_abs
        if math.isfinite(val):
            a = abs(val)
            if a < best_abs:
                best_x = x
                best_abs = a
                if best_abs <= 1e-8:
                    return True
        return False

    def safe_eval(x):
        try:
            v = wf(x)
            if not math.isfinite(v):
                return v, False
            return v, True
        except Exception:
            return float('nan'), False

    x0f = float(x0)
    base = max(1.0, abs(x0f))

    # --- Initial evaluation at x0 ---
    fx0, ok0 = safe_eval(x0f)
    if ok0 and try_update(x0f, fx0):
        return best_x

    # Cache to avoid re-evaluation
    cache = {x0f: (fx0, ok0)}
    
    # --- Phase 1: Generate 3 points for IQI ---
    # Point 1: x0
    # Point 2: x0 + h
    h = 1e-4 * base
    x1 = x0f + h
    fx1, ok1 = safe_eval(x1)
    if ok1 and try_update(x1, fx1):
        return best_x
    if not ok1:
        # Try other direction if it failed
        x1 = x0f - h
        fx1, ok1 = safe_eval(x1)
        if ok1 and try_update(x1, fx1):
            return best_x
        if not ok1:
            # Cannot proceed with IQI, go to bracketing
            pass

    # Point 3: Secant step from x0 and x1
    x2 = x0f
    fx2 = fx0
    x3 = x1
    fx3 = fx1
    
    if ok0 and ok1:
        denom = fx3 - fx2
        if abs(denom) > 1e-300:
            x_new = x3 - fx3 * (x3 - x2) / denom
            if math.isfinite(x_new):
                # Damp step if necessary
                step = x_new - x3
                max_step = 5.0 * max(1.0, abs(x3))
                if abs(step) > max_step:
                    x_new = x3 + max_step * math.copysign(1.0, step)
                
                fx_new, ok_new = safe_eval(x_new)
                if ok_new:
                    if try_update(x_new, fx_new):
                        return best_x
                    # We now have 3 points: x2, x3, x_new
                    # But we need them to be distinct and non-degenerate for IQI
                    # Let's use x2, x3, x_new as the initial triplet
                    x_a, f_a = x2, fx2
                    x_b, f_b = x3, fx3
                    x_c, f_c = x_new, fx_new
                else:
                    x_a, f_a = x2, fx2
                    x_b, f_b = x3, fx3
                    # Need a 3rd point. Try a small step from x3
                    x_c = x3 + 1e-5 * base
                    f_c, ok_c = safe_eval(x_c)
                    if not ok_c:
                        x_c = x3 - 1e-5 * base
                        f_c, ok_c = safe_eval(x_c)
                    if try_update(x_c, f_c):
                        return best_x
                    if not ok_c:
                        # Fallback to simple secant if IQI setup fails
                        x_a, f_a = x2, fx2
                        x_b, f_b = x3, fx3
                        x_c, f_c = None, None
            else:
                x_a, f_a = x2, fx2
                x_b, f_b = x3, fx3
                x_c, f_c = None, None
        else:
            x_a, f_a = x2, fx2
            x_b, f_b = x3, fx3
            x_c, f_c = None, None
    else:
        x_a, f_a = x2, fx2
        x_b, f_b = x3, fx3
        x_c, f_c = None, None

    # --- Phase 2: IQI/Secant Hybrid Loop ---
    # We maintain a list of the last 3 points
    # If x_c is None, we only have 2 points, so we must use Secant to get a 3rd
    if x_c is None:
        # Use Secant to get a 3rd point
        if ok0 and ok1:
            denom = fx3 - fx2
            if abs(denom) > 1e-300:
                x_new = x3 - fx3 * (x3 - x2) / denom
                if math.isfinite(x_new):
                    fx_new, ok_new = safe_eval(x_new)
                    if ok_new:
                        if try_update(x_new, fx_new):
                            return best_x
                        x_c, f_c = x_new, fx_new
                    else:
                        x_c, f_c = None, None
                else:
                    x_c, f_c = None, None
            else:
                x_c, f_c = None, None
        else:
            x_c, f_c = None, None

    if x_c is not None:
        pts = [(x_a, f_a), (x_b, f_b), (x_c, f_c)]
        
        for _ in range(30):
            if counter[0] >= max_calls or _time_left() < 5.0:
                break
            
            # Check if any point is a root
            for xp, fp in pts:
                if abs(fp) <= 1e-8:
                    return xp
            
            # Sort points by x to ensure distinctness? No, IQI works on any 3 distinct points.
            # But we need to ensure they are distinct.
            x1, f1 = pts[0]
            x2, f2 = pts[1]
            x3, f3 = pts[2]
            
            # Check for duplicate x or f
            if x1 == x2 or x2 == x3 or x1 == x3:
                # Duplicates exist, try to get a new point via Secant
                # Use the two most distinct points
                # This is a degenerate case, break and go to bracketing
                break
            
            denom1 = (f1 - f2) * (f1 - f3)
            denom2 = (f2 - f1) * (f2 - f3)
            denom3 = (f3 - f1) * (f3 - f2)
            
            if abs(denom1) < 1e-300 or abs(denom2) < 1e-300 or abs(denom3) < 1e-300:
                # Degenerate, use Secant
                # Use the two points with the largest difference in f
                if abs(f1 - f2) > abs(f2 - f3):
                    x_sec = x2 - f2 * (x2 - x1) / (f2 - f1)
                else:
                    x_sec = x3 - f3 * (x3 - x2) / (f3 - f2)
                
                if math.isfinite(x_sec):
                    f_sec, ok_sec = safe_eval(x_sec)
                    if ok_sec:
                        if try_update(x_sec, f_sec):
                            return best_x
                        # Update pts: keep the 3 most recent or most distinct?
                        # Keep the 3 points closest to the new point?
                        # Or just shift the window: (x2, x3, x_sec)
                        pts = [(x2, f2), (x3, f3), (x_sec, f_sec)]
                        continue
                break
            
            num1 = -f2 * -f3
            num2 = -f1 * -f3
            num3 = -f1 * -f2
            x_iqi = x1 * num1 / denom1 + x2 * num2 / denom2 + x3 * num3 / denom3
            
            if not math.isfinite(x_iqi):
                # Fallback to Secant
                x_sec = x3 - f3 * (x3 - x2) / (f3 - f2)
                if math.isfinite(x_sec):
                    f_sec, ok_sec = safe_eval(x_sec)
                    if ok_sec:
                        if try_update(x_sec, f_sec):
                            return best_x
                        pts = [(x2, f2), (x3, f3), (x_sec, f_sec)]
                        continue
                break
            
            # Stability check: x_iqi should be in the range of the two most recent x's
            # Let's assume x2 and x3 are the most recent.
            x_min = min(x2, x3)
            x_max = max(x2, x3)
            
            if x_min <= x_iqi <= x_max:
                x_new = x_iqi
            else:
                # Use Secant
                x_new = x3 - f3 * (x3 - x2) / (f3 - f2)
                if not math.isfinite(x_new):
                    break
            
            f_new, ok_new = safe_eval(x_new)
            if not ok_new:
                break
            
            if try_update(x_new, f_new):
                return best_x
            
            # Update pts: shift window
            pts = [(x2, f2), (x3, f3), (x_new, f_new)]

    # --- Phase 3: Global Bracketing ---
    # If we are here, local methods didn't converge.
    # We need to find a bracket [a, b] such that f(a)*f(b) < 0.
    
    # Strategy: Expand interval from x0 (or best_x)
    # Start with a small step, double it each time.
    # Check for sign change in [center - step, center + step]
    # If found, bisect or use Brent.
    
    center = best_x
    step = 1e-4 * max(1.0, abs(center))
    
    bracket_found = False
    a_br = None
    b_br = None
    fa_br = None
    fb_br = None
    
    for i in range(100): # Max 100 expansions
        if counter[0] >= max_calls or _time_left() < 3.0:
            break
        
        a = center - step
        b = center + step
        
        fa, oka = safe_eval(a)
        if oka and try_update(a, fa):
            return best_x
        
        fb, okb = safe_eval(b)
        if okb and try_update(b, fb):
            return best_x
        
        if oka and okb:
            if fa == 0.0:
                return a
            if fb == 0.0:
                return b
            if fa * fb < 0.0:
                a_br, b_br = a, b
                fa_br, fb_br = fa, fb
                bracket_found = True
                break
        
        # Expand step
        step *= 2.0
        
        # If step becomes very large, we might miss the root if it's in between.
        # But for oscillatory functions, sign changes are frequent.
        # If step is huge, we should probably check the middle.
        if step > 1e6:
            break

    if not bracket_found:
        # If no bracket found via expansion, try a more dense search around best_x
        # This is a fallback
        for scale in [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]:
            if counter[0] >= max_calls or _time_left() < 2.0:
                break
            h = scale * max(1.0, abs(best_x))
            # Sample 20 points in [best_x - h, best_x + h]
            n = 20
            prev_x = None
            prev_f = None
            for i in range(n + 1):
                x = best_x - h + 2 * h * i / n
                fx, ok = safe_eval(x)
                if ok:
                    if try_update(x, fx):
                        return best_x
                    if prev_x is not None:
                        if prev_f * fx < 0.0:
                            a_br, b_br = prev_x, x
                            fa_br, fb_br = prev_f, fx
                            bracket_found = True
                            break
                prev_x = x
                prev_f = fx
            if bracket_found:
                break

    # --- Phase 4: Brent's Method ---
    if bracket_found:
        try:
            res = brentq(wf, a_br, b_br, xtol=1e-14, rtol=1e-14, maxiter=50)
            if math.isfinite(res):
                fr, okr = safe_eval(res)
                if okr and try_update(res, fr):
                    return best_x
                if okr and abs(fr) < best_abs:
                    best_x = res
                    best_abs = abs(fr)
        except Exception:
            pass

    # --- Phase 5: Final Polish ---
    # If we still don't have a root, try a few more Secant steps from best_x
    if best_abs > 1e-8 and counter[0] < max_calls and _time_left() > 1.0:
        # Use best_x and a nearby point
        h = 1e-6 * max(1.0, abs(best_x))
        x1 = best_x
        f1 = None
        # Get f1 from cache if available
        for k, (v, ok) in cache.items():
            if k == x1 and ok:
                f1 = v
                break
        if f1 is None:
            f1, ok1 = safe_eval(x1)
            if not ok1:
                return best_x
        
        x2 = x1 + h
        f2, ok2 = safe_eval(x2)
        if not ok2:
            x2 = x1 - h
            f2, ok2 = safe_eval(x2)
        if not ok2:
            return best_x
        
        for _ in range(10):
            if counter[0] >= max_calls or _time_left() < 0.5:
                break
            if abs(f1) <= 1e-8:
                return x1
            if abs(f2) <= 1e-8:
                return x2
            denom = f2 - f1
            if abs(denom) < 1e-300:
                break
            x_new = x2 - f2 * (x2 - x1) / denom
            if not math.isfinite(x_new):
                break
            f_new, ok_new = safe_eval(x_new)
            if not ok_new:
                break
            if try_update(x_new, f_new):
                return best_x
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new

    return best_x
