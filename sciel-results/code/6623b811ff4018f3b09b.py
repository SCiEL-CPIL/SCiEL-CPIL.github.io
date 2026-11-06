"""Root finding: Ill-conditioned - Multi-strategy with minimization fallback."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Strategy:
    1. Secant method (efficient, robust)
    2. Newton with finite difference derivative
    3. Brent's method (if bracket found)
    4. Minimization of f(x)^2 (global-ish search)
    """
    start_time = time.time()
    MAX_WALL = 110.0
    MAX_EVALS = 500
    TARGET_TOL = 1e-8

    eval_count = [0]
    best_x = [float(x0)]
    best_f = [float('inf')]
    history_x = []
    history_f = []

    def eval_f(x):
        if eval_count[0] >= MAX_EVALS:
            raise RuntimeError("Max evaluations exceeded")
        if time.time() - start_time > MAX_WALL:
            raise RuntimeError("Time limit exceeded")
        
        eval_count[0] += 1
        try:
            val = f(x)
        except Exception:
            val = 0.0
        if not math.isfinite(val):
            val = 0.0
        
        if abs(val) < best_f[0]:
            best_f[0] = abs(val)
            best_x[0] = float(x)
        
        history_x.append(float(x))
        history_f.append(float(val))
        return val

    def is_good(x, fx):
        return abs(fx) <= TARGET_TOL

    def check_budget():
        return eval_count[0] < MAX_EVALS and (time.time() - start_time < MAX_WALL)

    # --- Strategy 1: Secant Method ---
    def try_secant(x0, max_iter=30):
        if not check_budget():
            return None
        
        x = float(x0)
        fx = eval_f(x)
        if is_good(x, fx):
            return x
        
        # Initialize second point with adaptive step
        h = 1e-4 * max(1.0, abs(x))
        x_prev = x + h
        fx_prev = eval_f(x_prev)
        if is_good(x_prev, fx_prev):
            return x_prev

        for _ in range(max_iter):
            if not check_budget():
                break
            if fx == fx_prev:
                break
            if abs(x - x_prev) < 1e-14 * max(1.0, abs(x)):
                break
                
            # Secant step
            d = (fx - fx_prev) / (x - x_prev)
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            
            fx_new = eval_f(x_new)
            if is_good(x_new, fx_new):
                return x_new
            
            x_prev, fx_prev = x, fx
            x, fx = x_new, fx_new
            
            if abs(x_new - x) < 1e-14 * max(1.0, abs(x_new)):
                break
        
        return x if abs(fx) <= TARGET_TOL else None

    # --- Strategy 2: Newton with 5-point FD ---
    def try_newton(x0, max_iter=30):
        if not check_budget():
            return None
        
        x = float(x0)
        fx = eval_f(x)
        if is_good(x, fx):
            return x
        
        for _ in range(max_iter):
            if not check_budget():
                break
            
            # 5-point central difference
            h = 1e-4 * max(1.0, abs(x))
            if h == 0: h = 1e-10
            
            fx_m2 = eval_f(x - 2*h)
            if is_good(x - 2*h, fx_m2): return x - 2*h
            fx_m1 = eval_f(x - h)
            if is_good(x - h, fx_m1): return x - h
            fx_p1 = eval_f(x + h)
            if is_good(x + h, fx_p1): return x + h
            fx_p2 = eval_f(x + 2*h)
            if is_good(x + 2*h, fx_p2): return x + 2*h
            
            d = (-fx_p2 + 8*fx_p1 - 8*fx_m1 + fx_m2) / (12*h)
            
            if d == 0 or not math.isfinite(d):
                break
                
            dx = fx / d
            if not math.isfinite(dx):
                break
            
            # Limit step to prevent divergence
            max_step = 1.0 * max(1.0, abs(x))
            if abs(dx) > max_step:
                dx = max_step * (1 if dx > 0 else -1)
                
            x_new = x - dx
            if not math.isfinite(x_new):
                break
                
            fx_new = eval_f(x_new)
            if is_good(x_new, fx_new):
                return x_new
                
            # Damping: if we got worse, halve the step
            if abs(fx_new) > abs(fx):
                improved = False
                for damp in [0.5, 0.25, 0.125]:
                    if not check_budget(): break
                    x_d = x - damp * dx
                    fx_d = eval_f(x_d)
                    if is_good(x_d, fx_d):
                        return x_d
                    if abs(fx_d) < abs(fx):
                        x, fx = x_d, fx_d
                        improved = True
                        break
                if not improved:
                    break
            else:
                x, fx = x_new, fx_new
                
            if abs(dx) < 1e-12 * max(1.0, abs(x)):
                break

        return x if abs(fx) <= TARGET_TOL else None

    # --- Strategy 3: Brent's Method (Fallback) ---
    def try_brent():
        if not check_budget():
            return None
            
        # Look for a sign change in history
        check_hist = history_x[-50:]
        check_f = history_f[-50:]
        
        bracket = None
        for i in range(len(check_hist) - 1):
            if check_f[i] * check_f[i+1] < 0:
                bracket = (check_hist[i], check_hist[i+1], check_f[i], check_f[i+1])
                break
        
        if bracket is None:
            # Try to find a bracket by scanning around best_x
            base_x = best_x[0]
            base_h = 1e-3 * max(1.0, abs(base_x))
            prev_x = base_x - base_h
            prev_fx = eval_f(prev_x)
            for i in range(1, 20):
                if not check_budget(): break
                cur_x = base_x + (i - 1) * base_h
                cur_fx = eval_f(cur_x)
                if is_good(cur_x, cur_fx):
                    return cur_x
                if prev_fx * cur_fx < 0:
                    bracket = (prev_x, cur_x, prev_fx, cur_fx)
                    break
                prev_x, prev_fx = cur_x, cur_fx
                
        if bracket is None:
            return None
            
        a, b, fa, fb = bracket
        try:
            from scipy.optimize import brentq
            def f_wrap(x):
                return eval_f(x)
            root = brentq(f_wrap, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            fr = eval_f(root)
            if is_good(root, fr):
                return root
        except Exception:
            pass
        return None

    # --- Strategy 4: Minimization of f(x)^2 ---
    def try_minimize():
        if not check_budget():
            return None
        
        try:
            from scipy.optimize import minimize_scalar
            
            def objective(x):
                fx = eval_f(x)
                return fx * fx
            
            # Define a reasonable search range around x0
            x0_val = float(x0)
            scale = max(1.0, abs(x0_val))
            bounds = (x0_val - 10 * scale, x0_val + 10 * scale)
            
            result = minimize_scalar(objective, bounds=bounds, method='bounded', 
                                     options={'xatol': 1e-12})
            
            if result.success:
                x_opt = result.x
                fx_opt = eval_f(x_opt)
                if is_good(x_opt, fx_opt):
                    return x_opt
                # Even if not perfectly good, return the best point found
                if abs(fx_opt) < best_f[0]:
                    return x_opt
        except Exception:
            pass
        return None

    # === MAIN EXECUTION ===
    
    def polish_root(x_start, fx_start=None, max_iter=20):
        """High-precision Newton polish to drive |f(x)| below 1e-8 reliably."""
        if not check_budget():
            return None
        x = float(x_start)
        if fx_start is not None:
            fx = float(fx_start)
        else:
            fx = eval_f(x)
            if is_good(x, fx):
                return x
        
        for _ in range(max_iter):
            if not check_budget():
                break
            if abs(fx) <= TARGET_TOL:
                return x
            
            # Adaptive h for derivative: balance truncation and roundoff error
            h = 1e-6 * max(1.0, abs(x))
            if h < 1e-12:
                h = 1e-12
            
            fx_p = eval_f(x + h)
            if is_good(x + h, fx_p):
                return x + h
            fx_m = eval_f(x - h)
            if is_good(x - h, fx_m):
                return x - h
            
            d = (fx_p - fx_m) / (2 * h)
            
            if d == 0 or not math.isfinite(d):
                break
            
            dx = fx / d
            if not math.isfinite(dx):
                break
            
            # Limit step size to avoid divergence
            max_step = 0.5 * max(1.0, abs(x))
            if abs(dx) > max_step:
                dx = max_step * (1 if dx > 0 else -1)
            
            x_new = x - dx
            if not math.isfinite(x_new):
                break
            
            fx_new = eval_f(x_new)
            if is_good(x_new, fx_new):
                return x_new
            
            # If step didn't improve, try smaller step
            if abs(fx_new) >= abs(fx):
                for damp in [0.5, 0.25, 0.1]:
                    if not check_budget():
                        break
                    x_d = x - damp * dx
                    fx_d = eval_f(x_d)
                    if is_good(x_d, fx_d):
                        return x_d
                    if abs(fx_d) < abs(fx):
                        x, fx = x_d, fx_d
                        break
                else:
                    break
            else:
                x, fx = x_new, fx_new
            
            if abs(dx) < 1e-14 * max(1.0, abs(x)):
                break
        
        return x if abs(fx) <= TARGET_TOL else None

    # 1. Try Secant (efficient, robust for many cases)
    result = try_secant(x0)
    if result is not None:
        # Polish the secant result for higher precision
        polished = polish_root(result)
        if polished is not None:
            return polished
        return result

    # 2. Try Newton from best point found so far
    result = try_newton(best_x[0])
    if result is not None:
        polished = polish_root(result)
        if polished is not None:
            return polished
        return result
        
    # 3. Try Brent
    result = try_brent()
    if result is not None:
        polished = polish_root(result)
        if polished is not None:
            return polished
        return result

    # 4. Try Minimization
    result = try_minimize()
    if result is not None:
        polished = polish_root(result)
        if polished is not None:
            return polished
        return result

    # 5. Final fallback: polish the best point found
    polished = polish_root(best_x[0])
    if polished is not None:
        return polished
    return best_x[0]

# EVOLVE-BLOCK-END
