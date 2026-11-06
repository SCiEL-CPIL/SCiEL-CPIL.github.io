"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 125.0  # Safety margin before 150s kill
    max_evals = 500
    TOL = 1e-8
    eval_count = 0
    best_x = float(x0)
    best_fabs = abs(f(best_x))
    eval_count += 1

    def call_f(x):
        nonlocal eval_count, best_x, best_fabs
        if eval_count >= max_evals:
            raise Exception("max evals")
        if time.time() > deadline:
            raise Exception("deadline")
        val = f(x)
        eval_count += 1
        if not (isinstance(val, (int, float)) and math.isfinite(val)):
            val = float('nan')
        fabs = abs(val)
        if math.isfinite(fabs) and fabs < best_fabs:
            best_fabs = fabs
            best_x = x
        return val

    def check_time():
        return time.time() < deadline

    def find_bracket(x0, max_evals_bracket=80):
        """Find a bracket [a, b] where f(a)*f(b) < 0 using adaptive expansion."""
        x = float(x0)
        try:
            fx = call_f(x)
        except Exception:
            return None
        if abs(fx) <= TOL:
            return (x, x, fx, fx)

        scale = max(1.0, abs(x))
        # Try multiple initial step sizes
        for base_step in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            step = base_step * scale
            a, fa = x, fx
            b, fb = None, None
            
            for direction in [1, -1]:
                step_dir = step
                for i in range(30):
                    if eval_count >= max_evals - 10:
                        return None
                    if not check_time():
                        return None
                    x_test = a + direction * step_dir
                    try:
                        f_test = call_f(x_test)
                    except Exception:
                        break
                    if not math.isfinite(f_test):
                        break
                    
                    if fa * f_test < 0:
                        if direction > 0:
                            return (a, x_test, fa, f_test)
                        else:
                            return (x_test, a, f_test, fa)
                    
                    # Move a forward if no sign change
                    a, fa = x_test, f_test
                    step_dir *= 2.0
                    if step_dir > 1e15 * scale:
                        break
        return None

    def inverse_cubic_interp(x1, f1, x2, f2, x3, f3, x4, f4):
        """Inverse cubic interpolation to find next point."""
        # Build Lagrange polynomial for x as function of f
        # Using divided differences
        def lagrange_eval(xi, points):
            result = 0.0
            for i in range(4):
                li = 1.0
                for j in range(4):
                    if i != j:
                        li *= (xi - points[j][1]) / (points[i][1] - points[j][1])
                result += points[i][0] * li
            return result

        points = [(x1, f1), (x2, f2), (x3, f3), (x4, f4)]
        # We want x such that f(x) = 0
        x_new = lagrange_eval(0.0, points)
        return x_new

    def try_hybrid_interp(x0, budget=200):
        """
        Hybrid approach:
        1. Find bracket if possible
        2. Use inverse cubic/quadratic interpolation within bracket
        3. Fall back to bisection if interpolation fails
        4. If no bracket, use secant with damping
        """
        x = float(x0)
        
        # First, try to find a bracket
        bracket = find_bracket(x, max_evals - eval_count - 20)
        
        if bracket is not None:
            a, b, fa, fb = bracket
            # Ensure |fa| <= |fb|
            if abs(fa) > abs(fb):
                a, b, fa, fb = b, a, fb, fa
            
            # If we already have a good point, use it
            if abs(fa) <= TOL:
                return True
            
            # Inverse interpolation within bracket
            # Keep last 3 points for cubic, 2 for quadratic
            history = [(a, fa), (b, fb)]
            
            for i in range(budget):
                if not check_time():
                    return False
                if eval_count >= max_evals - 5:
                    return False
                
                if abs(fb) <= TOL:
                    return True
                if abs(fa) <= TOL:
                    return True
                
                # Check if bracket is small enough
                if abs(b - a) < 1e-14 * max(1.0, abs(b)):
                    return min(abs(fa), abs(fb)) <= TOL
                
                # Choose interpolation method based on available points
                x_new = None
                
                if len(history) >= 3:
                    # Try inverse cubic interpolation
                    try:
                        h1, h2, h3, h4 = history[-1], history[-2], history[-3], history[-4] if len(history) >= 4 else history[-3]
                        if len(history) >= 4:
                            x_new = inverse_cubic_interp(h1[0], h1[1], h2[0], h2[1], h3[0], h3[1], h4[0], h4[1])
                        else:
                            # Quadratic interpolation
                            x1, f1 = history[-1]
                            x2, f2 = history[-2]
                            x3, f3 = history[-3]
                            # Lagrange quadratic
                            denom = (f1 - f2) * (f1 - f3) * (f2 - f3)
                            if abs(denom) > 1e-30:
                                x_new = (x1 * f2 * f3 / ((f1 - f2) * (f1 - f3)) + 
                                         x2 * f1 * f3 / ((f2 - f1) * (f2 - f3)) + 
                                         x3 * f1 * f2 / ((f3 - f1) * (f3 - f2)))
                    except Exception:
                        x_new = None
                
                if x_new is None or not math.isfinite(x_new):
                    # Fall back to secant (linear interpolation)
                    if abs(fb) < abs(fa):
                        x_new = b - fb * (b - a) / (fb - fa)
                    else:
                        x_new = a - fa * (b - a) / (fa - fb)
                
                # Ensure x_new is within bracket
                if not (min(a, b) - 1e-10 < x_new < max(a, b) + 1e-10):
                    # Bisection fallback
                    x_new = (a + b) / 2.0
                
                try:
                    f_new = call_f(x_new)
                except Exception:
                    return False
                
                if abs(f_new) <= TOL:
                    return True
                
                # Update bracket
                if fa * f_new < 0:
                    b, fb = x_new, f_new
                else:
                    a, fa = x_new, f_new
                
                # Ensure |fa| <= |fb|
                if abs(fa) > abs(fb):
                    a, b, fa, fb = b, a, fb, fa
                
                # Add to history (keep last 4)
                history.append((x_new, f_new))
                if len(history) > 4:
                    history.pop(0)
            
            return min(abs(fa), abs(fb)) <= TOL
        
        else:
            # No bracket found, use damped secant
            x1 = x
            try:
                f1 = call_f(x1)
            except Exception:
                return False
            if abs(f1) <= TOL:
                return True
            
            h = 1e-3 * max(1.0, abs(x1))
            x2 = x1 + h
            try:
                f2 = call_f(x2)
            except Exception:
                return False
            if abs(f2) <= TOL:
                return True
            
            for i in range(budget):
                if not check_time():
                    return False
                if eval_count >= max_evals - 5:
                    return False
                
                if abs(f2) <= TOL:
                    return True
                if abs(f1) <= TOL:
                    return True
                
                denom = f2 - f1
                if abs(denom) < 1e-30:
                    return False
                
                x_new = x2 - f2 * (x2 - x1) / denom
                if not math.isfinite(x_new):
                    return False
                if abs(x_new) > 1e15:
                    return False
                
                # Damp step
                max_step = 10.0 * max(1.0, abs(x2))
                if abs(x_new - x2) > max_step:
                    x_new = x2 + math.copysign(max_step, x_new - x2)
                
                x1, f1 = x2, f2
                x2 = x_new
                try:
                    f2 = call_f(x2)
                except Exception:
                    return False
            
            return abs(f2) <= TOL or abs(f1) <= TOL

    def try_secant_polish(x_start, tol=TOL, max_iter=40):
        """High-order secant polish with adaptive step size."""
        x = float(x_start)
        try:
            fx = call_f(x)
        except Exception:
            return False
        if abs(fx) <= tol:
            return True
        
        # Use multiple step sizes to get a good derivative estimate
        h1 = 1e-5 * max(1.0, abs(x))
        h2 = 1e-4 * max(1.0, abs(x))
        
        try:
            f_plus = call_f(x + h1)
            f_minus = call_f(x - h1)
        except Exception:
            return False
        
        if abs(f_plus) <= tol or abs(f_minus) <= tol:
            return True
        
        # Central difference for better derivative
        d = (f_plus - f_minus) / (2 * h1)
        if abs(d) < 1e-30:
            return False
        
        # Newton step
        step = fx / d
        max_step = 1.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = math.copysign(max_step, step)
        
        x_new = x - step
        if not math.isfinite(x_new) or abs(x_new) > 1e15:
            return False
        
        for i in range(max_iter):
            if not check_time():
                return False
            if eval_count >= max_evals - 5:
                return False
            
            try:
                fx_new = call_f(x_new)
            except Exception:
                return False
            
            if abs(fx_new) <= tol:
                return True
            
            # Update derivative estimate
            try:
                f_plus_new = call_f(x_new + h1)
                f_minus_new = call_f(x_new - h1)
            except Exception:
                return False
            
            d_new = (f_plus_new - f_minus_new) / (2 * h1)
            if abs(d_new) < 1e-30:
                # Fall back to secant
                if abs(fx_new - fx) < 1e-30:
                    return False
                step = fx_new * (x_new - x) / (fx_new - fx)
            else:
                step = fx_new / d_new
            
            if abs(step) > max_step:
                step = math.copysign(max_step, step)
            
            x = x_new
            fx = fx_new
            x_new = x - step
            
            if not math.isfinite(x_new) or abs(x_new) > 1e15:
                return False
        
        return abs(fx) <= tol

    def try_multi_start_secant(x0, budget=150):
        """Try secant from multiple perturbed starting points."""
        x = float(x0)
        scale = max(1.0, abs(x))
        
        # Try several offsets
        offsets = [0, 1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2, 1e-1, -1e-1]
        
        for offset in offsets:
            if eval_count >= max_evals - 10:
                break
            if not check_time():
                break
            
            x_start = x + offset * scale
            try:
                f1 = call_f(x_start)
            except Exception:
                continue
            
            if abs(f1) <= TOL:
                return True
            
            h = 1e-4 * max(1.0, abs(x_start))
            x2 = x_start + h
            try:
                f2 = call_f(x2)
            except Exception:
                continue
            
            if abs(f2) <= TOL:
                return True
            
            for i in range(30):
                if not check_time():
                    return False
                if eval_count >= max_evals - 5:
                    return False
                
                if abs(f2) <= TOL:
                    return True
                if abs(f1) <= TOL:
                    return True
                
                denom = f2 - f1
                if abs(denom) < 1e-30:
                    break
                
                x_new = x2 - f2 * (x2 - x_start) / denom
                if not math.isfinite(x_new) or abs(x_new) > 1e15:
                    break
                
                max_step = 10.0 * max(1.0, abs(x2))
                if abs(x_new - x2) > max_step:
                    x_new = x2 + math.copysign(max_step, x_new - x2)
                
                x_start, f1 = x2, f2
                x2 = x_new
                try:
                    f2 = call_f(x2)
                except Exception:
                    break
            
            if abs(f2) <= TOL or abs(f1) <= TOL:
                return True
        
        return False

    # Main strategy:
    # 1. Try hybrid interpolation (bracket + inverse cubic) - most robust
    # 2. If that fails, try multi-start secant
    # 3. Final polish from best point found

    solved = False
    
    # Phase 1: Hybrid interpolation (uses bracket if available)
    if try_hybrid_interp(x0, budget=250):
        solved = True
    
    # Phase 2: Multi-start secant if not solved
    if not solved and eval_count < max_evals - 30:
        if try_multi_start_secant(x0, budget=120):
            solved = True
    
    # Phase 3: Polish from best point
    if not solved and best_fabs < 1e-4 and eval_count < max_evals - 10:
        if try_secant_polish(best_x, TOL, 30):
            solved = True
    
    return best_x
# EVOLVE-BLOCK-END
