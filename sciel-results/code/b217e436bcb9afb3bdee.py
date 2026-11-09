"""Root finding for ill-conditioned equations: multi-strategy robust solver with enhanced techniques."""
# EVOLVE-BLOCK-START
import time
import math
import sys

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations of f.
    Returns the best x found (minimizing |f(x)|) within the budget.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    max_evals = 500
    evals_used = 0
    best_x = float(x0)
    best_fx = None
    best_abs = float('inf')
    
    def check_time():
        return time.time() < deadline
    
    def update_best(x, fx):
        nonlocal best_x, best_fx, best_abs
        if fx is not None and math.isfinite(fx):
            afx = abs(fx)
            if afx < best_abs:
                best_x = x
                best_fx = fx
                best_abs = afx
            return afx
        return None
    
    def safe_eval(x):
        nonlocal evals_used
        if not check_time():
            return None
        if evals_used >= max_evals:
            return None
        if not math.isfinite(x):
            return None
        try:
            val = f(x)
            evals_used += 1
            if isinstance(val, (int, float)) and math.isfinite(val):
                return float(val)
            return None
        except Exception:
            evals_used += 1
            return None
    
    def estimate_derivative(x, fx):
        """Estimate derivative using Richardson extrapolation for better accuracy."""
        h0 = 1e-7 * max(1.0, abs(x))
        # First level: central difference with h0
        fp1 = safe_eval(x + h0)
        if fp1 is None:
            return None
        fm1 = safe_eval(x - h0)
        if fm1 is None:
            # Forward difference
            return (fp1 - fx) / h0
        
        d1 = (fp1 - fm1) / (2.0 * h0)
        
        # Second level: central difference with h0/2
        h1 = h0 / 2.0
        fp2 = safe_eval(x + h1)
        if fp2 is None:
            return d1
        fm2 = safe_eval(x - h1)
        if fm2 is None:
            return d1
        
        d2 = (fp2 - fm2) / (2.0 * h1)
        
        # Richardson extrapolation: error ~ h^2, so d_ext = (4*d2 - d1)/3
        d_ext = (4.0 * d2 - d1) / 3.0
        
        if math.isfinite(d_ext) and abs(d_ext) > 1e-18:
            return d_ext
        elif math.isfinite(d2) and abs(d2) > 1e-18:
            return d2
        elif math.isfinite(d1) and abs(d1) > 1e-18:
            return d1
        return None
    
    def newton_step_with_damping(x, fx, df):
        """Newton step with backtracking line search for robustness."""
        if df is None or abs(df) < 1e-18:
            return None, None
        
        step = fx / df
        max_step = 20.0 * max(1.0, abs(x))
        
        if abs(step) > max_step:
            step = math.copysign(max_step, step)
        
        # Try full step first
        x_trial = x - step
        if not math.isfinite(x_trial):
            return None, None
        
        fx_trial = safe_eval(x_trial)
        if fx_trial is None:
            return None, None
        
        afx_trial = abs(fx_trial)
        update_best(x_trial, fx_trial)
        
        if afx_trial <= 1e-8:
            return x_trial, fx_trial
        
        # Backtracking if function value increased
        if afx_trial > abs(fx) * 1.1:
            for damp in [0.5, 0.25, 0.125, 0.0625, 0.03125]:
                x_trial = x - damp * step
                if not math.isfinite(x_trial):
                    continue
                fx_trial = safe_eval(x_trial)
                if fx_trial is None:
                    continue
                afx_trial = abs(fx_trial)
                update_best(x_trial, fx_trial)
                if afx_trial <= 1e-8:
                    return x_trial, fx_trial
                if afx_trial < abs(fx) * 0.9:
                    return x_trial, fx_trial
        
        return x_trial, fx_trial
    
    def bisection(a, b, fa, fb, max_iters=60):
        """Bisection method with convergence check."""
        if fa * fb > 0:
            return None
        
        for _ in range(max_iters):
            if not check_time() or evals_used >= max_evals - 2:
                break
            
            mid = 0.5 * (a + b)
            fm = safe_eval(mid)
            if fm is None:
                break
            
            update_best(mid, fm)
            
            if abs(fm) <= 1e-8:
                return mid
            
            if fa * fm <= 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
            
            if abs(b - a) < 1e-15 * max(1.0, abs(mid)):
                break
        
        # Return midpoint if converged
        mid = 0.5 * (a + b)
        fm = safe_eval(mid)
        if fm is not None:
            update_best(mid, fm)
            return mid
        return None
    
    # Initial evaluation with retries at different points
    fx = safe_eval(best_x)
    if fx is not None:
        update_best(best_x, fx)
    else:
        for dx in [1e-10, -1e-10, 1e-6, -1e-6, 1e-3, -1e-3, 1.0, -1.0, 10.0, -10.0, 100.0, -100.0]:
            if not check_time() or evals_used >= max_evals:
                break
            trial_x = float(x0) + dx
            trial_fx = safe_eval(trial_x)
            if trial_fx is not None:
                afx = update_best(trial_x, trial_fx)
                if afx is not None:
                    best_x = trial_x
                    best_fx = trial_fx
                    break
        if best_fx is None:
            return best_x
        if best_abs <= 1e-8:
            return best_x
    
    if best_abs <= 1e-8:
        return best_x
    
    # Strategy 1: Enhanced Newton with Richardson extrapolation and damping
    x = best_x
    fx = best_fx
    
    for it in range(35):
        if not check_time() or evals_used >= max_evals - 15:
            break
        if abs(fx) <= 1e-8:
            return x
        
        df = estimate_derivative(x, fx)
        if df is None:
            break
        
        x_new, fx_new = newton_step_with_damping(x, fx, df)
        if x_new is None or fx_new is None:
            break
        
        afx_new = abs(fx_new)
        
        if afx_new <= 1e-8:
            return x_new
        
        # Check convergence
        step = x - x_new
        if abs(step) < 1e-12 * max(1.0, abs(x_new)):
            break
        
        x = x_new
        fx = fx_new
    
    if best_abs <= 1e-8:
        return best_x
    
    # Strategy 2: Find a bracket and use bisection with enhanced search
    a = None
    b = None
    fa = None
    fb = None
    
    center = best_x
    base = max(1.0, abs(center))
    
    # More systematic bracket search with adaptive scales
    scales = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]
    
    for scale in scales:
        if not check_time() or evals_used >= max_evals - 25:
            break
        delta = scale * base
        if delta < 1e-15 * base:
            continue
        
        # Try multiple bracket configurations
        candidates = [
            (center - delta, center + delta),
            (center - 2*delta, center + 2*delta),
            (center, center + delta),
            (center - delta, center),
            (center - 5*delta, center + 5*delta),
            (center - 10*delta, center + 10*delta),
        ]
        
        for (xa, xb) in candidates:
            if not check_time() or evals_used >= max_evals - 20:
                break
            fa = safe_eval(xa)
            if fa is None:
                continue
            fb = safe_eval(xb)
            if fb is None:
                continue
            update_best(xa, fa)
            update_best(xb, fb)
            if best_abs <= 1e-8:
                return best_x
            if fa * fb < 0:
                a, b = xa, xb
                break
        if a is not None:
            break
    
    if a is not None and b is not None and check_time() and evals_used < max_evals - 10:
        result = bisection(a, b, fa, fb, max_iters=50)
        if result is not None and best_abs <= 1e-8:
            return best_x
    
    if best_abs <= 1e-8:
        return best_x
    
    # Strategy 3: Enhanced secant method with better initialization
    if check_time() and evals_used < max_evals - 15:
        x0s = best_x
        fx0 = best_fx
        
        # Try multiple initial offsets
        offsets = [1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4, 1e-3, -1e-3]
        x1s = None
        fx1 = None
        
        for offset in offsets:
            if not check_time() or evals_used >= max_evals - 10:
                break
            step_init = offset * max(1.0, abs(x0s))
            trial = x0s + step_init
            fx1 = safe_eval(trial)
            if fx1 is not None:
                update_best(trial, fx1)
                if best_abs <= 1e-8:
                    return best_x
                if abs(fx1 - fx0) > 1e-15:
                    x1s = trial
                    break
        
        if x1s is not None and fx1 is not None:
            for _ in range(30):
                if not check_time() or evals_used >= max_evals - 3:
                    break
                if abs(fx1) < abs(fx0):
                    x0s, x1s = x1s, x0s
                    fx0, fx1 = fx1, fx0
                if abs(fx1) <= 1e-8:
                    return x1s
                denom = fx1 - fx0
                if abs(denom) < 1e-16:
                    break
                x2s = x1s - fx1 * (x1s - x0s) / denom
                if not math.isfinite(x2s):
                    break
                max_step = 15.0 * max(1.0, abs(x1s))
                if abs(x2s - x1s) > max_step:
                    x2s = x1s - math.copysign(max_step, x2s - x1s)
                fx2 = safe_eval(x2s)
                if fx2 is None:
                    break
                update_best(x2s, fx2)
                if abs(fx2) <= 1e-8:
                    return x2s
                x0s, x1s = x1s, x2s
                fx0, fx1 = fx1, fx2
    
    # Strategy 4: Polishing with higher-order methods when close
    if check_time() and evals_used < max_evals - 5 and best_abs < 1e-4:
        # Try a few Newton-like steps with very small h for derivative
        x_polish = best_x
        fx_polish = best_fx
        
        for _ in range(5):
            if not check_time() or evals_used >= max_evals - 2:
                break
            if abs(fx_polish) <= 1e-8:
                return x_polish
            
            h = 1e-10 * max(1.0, abs(x_polish))
            fp = safe_eval(x_polish + h)
            if fp is None:
                break
            fm = safe_eval(x_polish - h)
            if fm is not None:
                df = (fp - fm) / (2.0 * h)
            else:
                df = (fp - fx_polish) / h
            
            if not math.isfinite(df) or abs(df) < 1e-18:
                break
            
            step = fx_polish / df
            if abs(step) < 1e-14:
                break
            
            x_new = x_polish - step
            if not math.isfinite(x_new):
                break
            
            fx_new = safe_eval(x_new)
            if fx_new is None:
                break
            
            update_best(x_new, fx_new)
            
            if abs(fx_new) <= 1e-8:
                return x_new
            
            if abs(fx_new) < abs(fx_polish) * 0.5:
                x_polish = x_new
                fx_polish = fx_new
            else:
                break
    
    # Strategy 5: Final grid search if still stuck
    if check_time() and evals_used < max_evals - 5 and best_abs > 1e-10:
        center = best_x
        base = max(1.0, abs(center))
        for power in range(-16, 4):
            if not check_time() or evals_used >= max_evals - 3:
                break
            delta = (10 ** power) * base
            if delta < 1e-16 * base:
                continue
            for sign in [1, -1]:
                if not check_time() or evals_used >= max_evals - 3:
                    break
                trial = center + sign * delta
                fval = safe_eval(trial)
                if fval is not None:
                    update_best(trial, fval)
                    if best_abs <= 1e-8:
                        return best_x
    
    return best_x
# EVOLVE-BLOCK-END
