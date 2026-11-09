"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses inverse-secant interpolation with history reuse for efficiency.
    """
    x0 = float(x0)
    start_time = time.time()
    DEADLINE = 145.0  # Leave 5s safety margin
    MAX_EVALS = 490   # Leave margin under 500
    
    eval_count = [0]
    
    def eval_f(x):
        if eval_count[0] >= MAX_EVALS:
            raise ValueError("Max evaluations exceeded")
        if time.time() - start_time > DEADLINE:
            raise ValueError("Time limit exceeded")
        eval_count[0] += 1
        return f(x)
    
    best_x = x0
    best_f = abs(eval_f(x0))
    
    def update_best(x, fx):
        nonlocal best_x, best_f
        afx = abs(fx)
        if afx < best_f:
            best_x = x
            best_f = afx
        if afx <= 1e-8:
            return True
        return False
    
    # Check if x0 is already a root
    if best_f <= 1e-8:
        return best_x
    
    # Inverse-secant interpolation using history of (x, f) pairs
    def inverse_secant_interp(x_history, f_history, max_iter=100):
        """
        Use inverse interpolation on history to predict next x.
        Requires 1 eval per iteration after initial setup.
        """
        if len(x_history) < 2:
            return None
        
        # Use the two most recent points for secant
        x1, x2 = x_history[-1], x_history[-2]
        f1, f2 = f_history[-1], f_history[-2]
        
        if abs(f1 - f2) < 1e-15:
            return None
        
        # Secant step
        x_new = x1 - f1 * (x1 - x2) / (f1 - f2)
        
        if not math.isfinite(x_new):
            return None
        
        return x_new
    
    # Main solver using inverse-secant with history
    def solve_with_history(x_start, max_iter=150):
        x = float(x_start)
        fx = eval_f(x)
        update_best(x, fx)
        if abs(fx) <= 1e-8:
            return x
        
        # Build initial history with a second point
        h = 1e-4 * max(1.0, abs(x))
        x2 = x + h
        fx2 = eval_f(x2)
        update_best(x2, fx2)
        if abs(fx2) <= 1e-8:
            return x2
        
        # History: most recent first
        x_hist = [x, x2]
        f_hist = [fx, fx2]
        
        for i in range(max_iter):
            if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
                break
            
            # Get next point from inverse interpolation
            x_new = inverse_secant_interp(x_hist, f_hist)
            
            if x_new is None:
                # Fallback: try a different perturbation
                perturb = (1e-3 + 1e-5 * i) * max(1.0, abs(x_hist[0])) * (1 if i % 2 == 0 else -1)
                x_new = x_hist[0] + perturb
            
            if not math.isfinite(x_new):
                break
            
            # Check if too close to previous points
            if any(abs(x_new - xh) < 1e-14 * max(1.0, abs(xh)) for xh in x_hist[:5]):
                # Try a different perturbation
                perturb = (1e-2 + 1e-4 * i) * max(1.0, abs(x_hist[0])) * (1 if i % 2 == 0 else -1)
                x_new = x_hist[0] + perturb
                if not math.isfinite(x_new):
                    break
            
            fx_new = eval_f(x_new)
            if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
                break
            
            if update_best(x_new, fx_new):
                return x_new
            
            # Update history (keep most recent 10 points)
            x_hist.insert(0, x_new)
            f_hist.insert(0, fx_new)
            if len(x_hist) > 10:
                x_hist.pop()
                f_hist.pop()
        
        return x_hist[0]
    
    # Strategy 1: Inverse-secant with history from x0
    x_result = solve_with_history(x0, max_iter=120)
    if best_f <= 1e-8:
        return best_x
    
    # Strategy 2: Try from perturbed starting points
    for perturb in [1e-2, -1e-2, 1e-3, -1e-3, 1e-4, -1e-4]:
        if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
            break
        x_pert = x0 + perturb * max(1.0, abs(x0))
        x_result = solve_with_history(x_pert, max_iter=60)
        if best_f <= 1e-8:
            return best_x
    
    # Strategy 3: Bisection if we can find a bracket
    def find_bracket(x0_val, max_expansions=20):
        a = float(x0_val)
        fa = eval_f(a)
        if abs(fa) <= 1e-8:
            return (a, a)
        
        step = 1.0
        for _ in range(max_expansions):
            if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
                break
            
            b = a + step
            fb = eval_f(b)
            if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
                break
            if abs(fb) <= 1e-8:
                return (b, b)
            if fa * fb < 0:
                return (a, b)
            
            c = a - step
            fc = eval_f(c)
            if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
                break
            if abs(fc) <= 1e-8:
                return (c, c)
            if fa * fc < 0:
                return (c, a)
            
            step *= 2.0
            if step > 1e10:
                break
        
        return None
    
    def bisection(a, b, max_iter=60):
        fa = eval_f(a)
        fb = eval_f(b)
        update_best(a, fa)
        update_best(b, fb)
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b
        
        for i in range(max_iter):
            if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
                break
            
            if abs(b - a) < 1e-14:
                break
            
            mid = 0.5 * (a + b)
            fm = eval_f(mid)
            if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
                break
            
            if update_best(mid, fm):
                return mid
            
            if fa * fm <= 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
        
        return 0.5 * (a + b)
    
    bracket = find_bracket(x0, max_expansions=15)
    if bracket is not None:
        a, b = bracket
        if a != b:
            x_result = bisection(a, b, max_iter=50)
            if best_f <= 1e-8:
                return best_x
    
    # Return best result found
    return best_x
# EVOLVE-BLOCK-END
