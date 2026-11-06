"""Root finding: Ill-conditioned - robust solver with multiple strategies."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a portfolio approach: Newton, secant, bisection with bracketing search.
    """
    start_time = time.time()
    MAX_WALL = 115.0  # Safety margin before 120s limit
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    eval_count = [0]
    
    def eval_f(x):
        """Counted evaluation of f."""
        eval_count[0] += 1
        return f(x)
    
    def time_ok():
        return time.time() - start_time < MAX_WALL
    
    def newton_step(x, fx, h=None):
        """
        One Newton step with finite difference derivative.
        Returns (x_new, fx_new, success)
        """
        if not time_ok() or eval_count[0] >= MAX_EVALS - 2:
            return x, fx, False
        
        if h is None:
            h = 1e-5 * max(1.0, abs(x))
        
        if eval_count[0] >= MAX_EVALS - 1:
            return x, fx, False
        
        fx_plus = eval_f(x + h)
        d = (fx_plus - fx) / h
        
        if d == 0 or not math.isfinite(d):
            # Try larger h
            h2 = h * 10
            if eval_count[0] >= MAX_EVALS - 1:
                return x, fx, False
            fx_plus2 = eval_f(x + h2)
            d = (fx_plus2 - fx) / h2
            if d == 0 or not math.isfinite(d):
                return x, fx, False
        
        step = fx / d
        if not math.isfinite(step):
            return x, fx, False
        
        # Damp the step if too large
        if abs(step) > 10 * max(1.0, abs(x)):
            step = math.copysign(10 * max(1.0, abs(x)), step)
        
        x_new = x - step
        if not math.isfinite(x_new):
            return x, fx, False
        
        if eval_count[0] >= MAX_EVALS:
            return x, fx, False
        
        fx_new = eval_f(x_new)
        return x_new, fx_new, True
    
    def try_newton(x_start, fx_start, max_iters=50, h_base=1e-5):
        """Newton's method with damping and adaptive step size."""
        x = float(x_start)
        fx = fx_start
        h = h_base * max(1.0, abs(x))
        prev_fx = None
        
        for _ in range(max_iters):
            if abs(fx) <= TARGET_TOL:
                return x, fx, True
            if not time_ok() or eval_count[0] >= MAX_EVALS - 2:
                break
            
            x_new, fx_new, ok = newton_step(x, fx, h)
            if not ok:
                # Try with a smaller step
                h = h * 0.1
                if h < 1e-15:
                    break
                x_new, fx_new, ok = newton_step(x, fx, h)
                if not ok:
                    break
            
            # Check if we're making progress
            if prev_fx is not None and abs(fx_new) > abs(prev_fx) * 1.01:
                # Step was too large, reduce h
                h = h * 0.1
            else:
                # Step was good, can increase h slightly
                h = min(h * 1.5, 1e-2 * max(1.0, abs(x)))
            
            prev_fx = abs(fx)
            x, fx = x_new, fx_new
        
        return x, fx, abs(fx) <= TARGET_TOL
    
    def try_secant(x1, fx1, x2, fx2, max_iters=50):
        """Secant method."""
        for _ in range(max_iters):
            if abs(fx1) <= TARGET_TOL:
                return x1, fx1, True
            if abs(fx2) <= TARGET_TOL:
                return x2, fx2, True
            if not time_ok() or eval_count[0] >= MAX_EVALS:
                break
            
            if fx1 == fx2 or x1 == x2:
                break
            
            d = (fx2 - fx1) / (x2 - x1)
            if d == 0 or not math.isfinite(d):
                break
            
            x_new = x1 - fx1 / d
            if not math.isfinite(x_new):
                break
            
            x1, fx1 = x2, fx2
            if eval_count[0] >= MAX_EVALS:
                break
            x2, fx2 = x_new, eval_f(x_new)
        
        best_x = x1 if abs(fx1) <= abs(fx2) else x2
        best_fx = fx1 if abs(fx1) <= abs(fx2) else fx2
        return best_x, best_fx, abs(best_fx) <= TARGET_TOL
    
    def bisection(a, fa, b, fb, max_iters=60):
        """Bisection method. Requires sign change."""
        if fa * fb > 0:
            return a if abs(fa) < abs(fb) else b, min(abs(fa), abs(fb)), False
        
        for _ in range(max_iters):
            if abs(fa) <= TARGET_TOL:
                return a, fa, True
            if abs(fb) <= TARGET_TOL:
                return b, fb, True
            if not time_ok() or eval_count[0] >= MAX_EVALS:
                break
            
            mid = (a + b) / 2
            if mid == a or mid == b:
                break
            
            fm = eval_f(mid)
            if abs(fm) <= TARGET_TOL:
                return mid, fm, True
            
            if fa * fm <= 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
        
        best_x = a if abs(fa) <= abs(fb) else b
        best_fx = fa if abs(fa) <= abs(fb) else fb
        return best_x, best_fx, abs(best_fx) <= TARGET_TOL
    
    def try_illinois(a, fa, b, fb, max_iters=50):
        """Illinois method (modified bisection with linear interpolation)."""
        if fa * fb > 0:
            return a if abs(fa) < abs(fb) else b, min(abs(fa), abs(fb)), False
        
        for _ in range(max_iters):
            if abs(fa) <= TARGET_TOL:
                return a, fa, True
            if abs(fb) <= TARGET_TOL:
                return b, fb, True
            if not time_ok() or eval_count[0] >= MAX_EVALS:
                break
            
            denom = fb - fa
            if denom == 0:
                x_mid = (a + b) / 2
            else:
                x_mid = a - fa * (b - a) / denom
            
            if x_mid <= a or x_mid >= b:
                x_mid = (a + b) / 2
            
            if x_mid == a or x_mid == b:
                break
            
            fm = eval_f(x_mid)
            if abs(fm) <= TARGET_TOL:
                return x_mid, fm, True
            
            if fa * fm <= 0:
                b, fb = x_mid, fm
                fa = 0.5 * fa
            else:
                a, fa = x_mid, fm
                fb = 0.5 * fb
        
        best_x = a if abs(fa) <= abs(fb) else b
        best_fx = fa if abs(fa) <= abs(fb) else fb
        return best_x, best_fx, abs(best_fx) <= TARGET_TOL
    
    # --- Main logic ---
    
    x0 = float(x0)
    fx0 = eval_f(x0)
    
    if abs(fx0) <= TARGET_TOL:
        return x0
    
    best_x = x0
    best_fx = fx0
    
    # Strategy 1: Newton from x0
    if time_ok() and eval_count[0] < MAX_EVALS - 5:
        x_n, fx_n, ok = try_newton(x0, fx0, max_iters=40, h_base=1e-5)
        if ok:
            return x_n
        if abs(fx_n) < abs(best_fx):
            best_x, best_fx = x_n, fx_n
    
    # Strategy 2: Secant from x0 and nearby point
    if time_ok() and eval_count[0] < MAX_EVALS - 10:
        h_sec = 1e-4 * max(1.0, abs(x0))
        x2 = x0 + h_sec
        if x2 != x0:
            fx2 = eval_f(x2)
            if abs(fx2) <= TARGET_TOL:
                return x2
            if abs(fx2) < abs(best_fx):
                best_x, best_fx = x2, fx2
            
            x_s, fx_s, ok = try_secant(x0, fx0, x2, fx2, max_iters=40)
            if ok:
                return x_s
            if abs(fx_s) < abs(best_fx):
                best_x, best_fx = x_s, fx_s
    
    # Strategy 3: Search for bracket and use bisection/Illinois
    if time_ok() and eval_count[0] < MAX_EVALS - 30:
        # Try to find a sign change by expanding from x0
        base_h = max(1e-6, abs(x0) * 1e-6)
        
        for scale in [1, 10, 100, 1000, 10000, 100000]:
            if not time_ok() or eval_count[0] >= MAX_EVALS - 10:
                break
            h = base_h * scale
            if h < 1e-15:
                continue
            
            for sign in [1, -1]:
                if not time_ok() or eval_count[0] >= MAX_EVALS - 10:
                    break
                x_c = x0 + sign * h
                if x_c == x0:
                    continue
                fx_c = eval_f(x_c)
                
                if abs(fx_c) <= TARGET_TOL:
                    return x_c
                if abs(fx_c) < abs(best_fx):
                    best_x, best_fx = x_c, fx_c
                
                if fx0 * fx_c < 0:
                    # Found bracket
                    a, b = (x0, x_c) if x0 < x_c else (x_c, x0)
                    fa, fb = (fx0, fx_c) if x0 < x_c else (fx_c, fx0)
                    
                    # Try Illinois first (faster convergence near root)
                    x_i, fx_i, ok = try_illinois(a, fa, b, fb, max_iters=50)
                    if ok:
                        return x_i
                    if abs(fx_i) < abs(best_fx):
                        best_x, best_fx = x_i, fx_i
                    
                    # Then bisection for guaranteed convergence
                    x_b, fx_b, ok = bisection(a, fa, b, fb, max_iters=50)
                    if ok:
                        return x_b
                    if abs(fx_b) < abs(best_fx):
                        best_x, best_fx = x_b, fx_b
                    break
    
    # Strategy 4: Try Newton with different step sizes
    for h_base in [1e-3, 1e-2, 1e-4, 1e-6, 1e-7]:
        if not time_ok() or eval_count[0] >= MAX_EVALS - 5:
            break
        x_n, fx_n, ok = try_newton(x0, fx0, max_iters=30, h_base=h_base)
        if ok:
            return x_n
        if abs(fx_n) < abs(best_fx):
            best_x, best_fx = x_n, fx_n
    
    # Strategy 4b: Final high-precision polish pass on best_x
    if time_ok() and eval_count[0] < MAX_EVALS - 10 and abs(best_fx) > 0:
        # Try to polish the best point found so far
        for h_base in [1e-6, 1e-7, 1e-8]:
            if not time_ok() or eval_count[0] >= MAX_EVALS - 5:
                break
            x_p, fx_p, ok = try_newton(best_x, best_fx, max_iters=20, h_base=h_base)
            if ok:
                return x_p
            if abs(fx_p) < abs(best_fx):
                best_x, best_fx = x_p, fx_p
    
    # Strategy 5: Try from perturbed starting points
    if time_ok() and eval_count[0] < MAX_EVALS - 20:
        for perturb in [1e-2, -1e-2, 1e-1, -1e-1, 1.0, -1.0]:
            if not time_ok() or eval_count[0] >= MAX_EVALS - 10:
                break
            x_p = x0 + perturb * max(1.0, abs(x0))
            if x_p == x0:
                continue
            fx_p = eval_f(x_p)
            if abs(fx_p) <= TARGET_TOL:
                return x_p
            if abs(fx_p) < abs(best_fx):
                best_x, best_fx = x_p, fx_p
            
            x_n, fx_n, ok = try_newton(x_p, fx_p, max_iters=20, h_base=1e-4)
            if ok:
                return x_n
            if abs(fx_n) < abs(best_fx):
                best_x, best_fx = x_n, fx_n
    
    # Strategy 5b: Try secant from best_x with a nearby point
    if time_ok() and eval_count[0] < MAX_EVALS - 10:
        h = 1e-5 * max(1.0, abs(best_x))
        x2 = best_x + h
        if x2 != best_x:
            fx2 = eval_f(x2)
            if abs(fx2) <= TARGET_TOL:
                return x2
            if abs(fx2) < abs(best_fx):
                best_x, best_fx = x2, fx2
            
            x_s, fx_s, ok = try_secant(best_x, best_fx, x2, fx2, max_iters=30)
            if ok:
                return x_s
            if abs(fx_s) < abs(best_fx):
                best_x, best_fx = x_s, fx_s
    
    # Strategy 6: Final secant attempts with best points
    if time_ok() and eval_count[0] < MAX_EVALS - 10:
        h = 1e-3 * max(1.0, abs(best_x))
        x2 = best_x + h
        if x2 != best_x:
            fx2 = eval_f(x2)
            if abs(fx2) <= TARGET_TOL:
                return x2
            x_s, fx_s, ok = try_secant(best_x, best_fx, x2, fx2, max_iters=30)
            if ok:
                return x_s
            if abs(fx_s) < abs(best_fx):
                best_x, best_fx = x_s, fx_s
    
    # Strategy 7: Final Illinois/bisection on best bracket if one was found
    if time_ok() and eval_count[0] < MAX_EVALS - 5:
        # Try to find a tight bracket around best_x
        for h_scale in [1e-2, 1e-3, 1e-4]:
            if not time_ok() or eval_count[0] >= MAX_EVALS - 2:
                break
            h = h_scale * max(1.0, abs(best_x))
            x_a = best_x - h
            x_b = best_x + h
            if x_a == x_b or x_a == best_x or x_b == best_x:
                continue
            try:
                fa = eval_f(x_a)
                fb = eval_f(x_b)
            except Exception:
                continue
            
            if abs(fa) <= TARGET_TOL:
                return x_a
            if abs(fb) <= TARGET_TOL:
                return x_b
            if abs(fa) < abs(best_fx):
                best_x, best_fx = x_a, fa
            if abs(fb) < abs(best_fx):
                best_x, best_fx = x_b, fb
            
            if fa * fb < 0:
                x_i, fx_i, ok = try_illinois(x_a, fa, x_b, fb, max_iters=30)
                if ok:
                    return x_i
                if abs(fx_i) < abs(best_fx):
                    best_x, best_fx = x_i, fx_i
    
    # Return best found
    return best_x

# EVOLVE-BLOCK-END
