import math
import time
import numpy as np
from scipy.optimize import brentq, newton, secant

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    x = float(x0)
    best_x = x
    best_fx = f(x)
    best_abs_fx = abs(best_fx)
    
    if best_abs_fx <= 1e-8:
        return best_x

    # Helper to check validity
    def is_valid(val):
        return not (math.isnan(val) or math.isinf(val))

    # Strategy: 
    # 1. Try fast, robust global bracketing search.
    # 2. If bracket found, use Brent's method (guaranteed convergence, efficient).
    # 3. If no bracket, try Newton with adaptive step and fallbacks.
    # 4. If Newton fails, try Secant or other local methods.
    
    # --- Phase 1: Global Search for a Bracket ---
    # We expand a window around x0 to find a sign change.
    # This is robust for many ill-conditioned problems where Newton might jump away.
    
    base_delta = 1e-3 * max(1.0, abs(x))
    if base_delta == 0:
        base_delta = 1e-3
        
    found_bracket = False
    a, b, fa, fb = None, None, None, None
    
    # Try expanding brackets
    for i in range(20):
        if time.time() > deadline:
            break
            
        delta = base_delta * (10 ** i)
        # Limit delta to avoid overflow or going too far
        if delta > 1e10 * max(1.0, abs(x)):
            break
            
        left = x - delta
        right = x + delta
        
        fl = f(left)
        fr = f(right)
        
        if not (is_valid(fl) and is_valid(fr)):
            continue
            
        if abs(fl) <= 1e-8:
            best_x = left
            best_fx = fl
            best_abs_fx = abs(fl)
            return best_x
            
        if abs(fr) <= 1e-8:
            best_x = right
            best_fx = fr
            best_abs_fx = abs(fr)
            return best_x
            
        if fl * fr < 0:
            a, b = left, right
            fa, fb = fl, fr
            found_bracket = True
            break
            
        # Update best if closer
        if abs(fl) < best_abs_fx:
            best_x = left
            best_fx = fl
            best_abs_fx = abs(fl)
        if abs(fr) < best_abs_fx:
            best_x = right
            best_fx = fr
            best_abs_fx = abs(fr)

    if found_bracket:
        # Use Brent's method on the bracket
        # Brent is guaranteed to converge and usually takes < 10-20 evals
        try:
            # maxiter is a safety, rtol controls precision
            root = brentq(f, a, b, maxiter=100, rtol=1e-12, xtol=1e-14)
            fr = f(root)
            if is_valid(fr):
                if abs(fr) <= 1e-8:
                    return root
                elif abs(fr) < best_abs_fx:
                    best_x = root
                    best_fx = fr
                    best_abs_fx = abs(fr)
            return best_x
        except Exception:
            # If Brent fails, fall through to local methods
            pass

    # --- Phase 2: Local Methods (Newton/Secant) ---
    # If no global bracket was found, the function might not change sign near x0,
    # or the root is a double root (tangency).
    
    # Try Newton with finite difference derivative
    # We use a small step for derivative, but damp the actual step to avoid divergence.
    
    h = 1e-5 * max(1.0, abs(x))
    x_current = x
    fx_current = best_fx
    
    for i in range(100):
        if time.time() > deadline:
            break
            
        if best_abs_fx <= 1e-8:
            return best_x
            
        # Calculate derivative
        # Central difference is better but costs 2 evals. Forward costs 1 (reusing fx).
        # Let's use forward difference for efficiency, but be careful with h.
        h = 1e-5 * max(1.0, abs(x_current))
        if h == 0:
            h = 1e-5
            
        fx_h = f(x_current + h)
        if not is_valid(fx_h):
            # Try smaller h
            h *= 0.1
            fx_h = f(x_current + h)
            if not is_valid(fx_h):
                break
                
        df = (fx_h - fx_current) / h
        
        if not is_valid(df) or abs(df) < 1e-15:
            # Derivative is too small or invalid.
            # Try Secant method using previous point? 
            # Or just perturb and restart Newton?
            # Let's try a small random perturbation to escape flat region
            x_current += 1e-5 * max(1.0, abs(x_current))
            fx_current = f(x_current)
            if not is_valid(fx_current):
                break
            if abs(fx_current) < best_abs_fx:
                best_x = x_current
                best_fx = fx_current
                best_abs_fx = abs(fx_current)
            continue
            
        # Newton step
        step = fx_current / df
        x_new = x_current - step
        
        # Damping: limit step size to prevent flying to infinity
        max_step = 1.0 + abs(x_current)
        if abs(step) > max_step:
            step = max_step * (1.0 if step > 0 else -1.0)
            x_new = x_current - step
            
        fx_new = f(x_new)
        
        if not is_valid(fx_new):
            # Step went to bad territory. Shrink step.
            step *= 0.5
            x_new = x_current - step
            fx_new = f(x_new)
            if not is_valid(fx_new):
                break
                
        if abs(fx_new) < best_abs_fx:
            # Accept step
            x_current = x_new
            fx_current = fx_new
            best_x = x_new
            best_fx = fx_new
            best_abs_fx = abs(fx_new)
            
            if best_abs_fx <= 1e-8:
                return best_x
        else:
            # Reject step. Try a damped version immediately.
            # If even damped doesn't improve, we might be stuck.
            # Try half step
            step_half = step * 0.5
            x_half = x_current - step_half
            fx_half = f(x_half)
            
            if is_valid(fx_half) and abs(fx_half) < best_abs_fx:
                x_current = x_half
                fx_current = fx_half
                best_x = x_half
                best_fx = fx_half
                best_abs_fx = abs(fx_half)
                
                if best_abs_fx <= 1e-8:
                    return best_x
            else:
                # If we can't improve with Newton, try Secant with a different point
                # Or break and return best
                # Let's try one more thing: if we are close to a root but f is flat,
                # maybe we need to move further?
                # For now, break to avoid wasting evals.
                break

    return best_x
