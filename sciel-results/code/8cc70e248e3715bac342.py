"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned problems.
    Uses a combination of:
    1. Safeguarded Inverse Quadratic Interpolation (SIQI) with actual quadratic interp
    2. Newton's method with adaptive damping
    3. Secant method fallback
    4. Bisection for guaranteed convergence
    5. Multiple restarts if initial guess is poor
    6. Final high-precision polish pass
    7. Deadline guard to stay within time limits
    """
    deadline = time.time() + 140  # Leave 10s margin before 150s kill
    max_evals = 450
    eval_count = [0]
    
    def eval_f(x):
        if time.time() > deadline:
            return None
        if eval_count[0] >= max_evals:
            return None
        try:
            val = f(x)
            eval_count[0] += 1
            if not math.isfinite(val):
                return None
            return val
        except (ValueError, OverflowError, ZeroDivisionError, TypeError):
            eval_count[0] += 1
            return None
    
    x = float(x0)
    fx = eval_f(x)
    if fx is None:
        return x
    if abs(fx) <= 1e-8:
        return x
    
    # Track best solution found so far
    best_x = x
    best_fx = fx
    
    # Safeguarded Inverse Quadratic Interpolation (SIQI)
    def try_siqi(x_start, fx_start, max_iter=40):
        """
        SIQI: Uses quadratic interpolation on (x, f(x)) with safeguarding.
        Maintains a bracket [a, b] and uses interpolation to pick the next point,
        falling back to bisection if interpolation goes outside the bracket.
        """
        # Initialize bracket around x_start
        h0 = 1e-4 * max(1.0, abs(x_start))
        a = x_start - h0
        b = x_start + h0
        
        fa = eval_f(a)
        fb = eval_f(b)
        
        # If we already have a good point, try to use it
        if fa is None or fb is None:
            # Try different bracket sizes
            for mult in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
                h = mult * max(1.0, abs(x_start))
                a = x_start - h
                b = x_start + h
                fa = eval_f(a)
                fb = eval_f(b)
                if fa is not None and fb is not None:
                    break
            if fa is None or fb is None:
                return None
        
        # Ensure we have a sign change
        if fa * fb > 0:
            # No sign change, try expanding
            step = max(1.0, abs(x_start)) * 0.5
            for _ in range(20):
                if time.time() > deadline:
                    return None
                a_new = a - step
                fa_new = eval_f(a_new)
                if fa_new is not None and fa_new * fb < 0:
                    a, fa = a_new, fa_new
                    break
                if fa_new is not None:
                    a, fa = a_new, fa_new
                
                b_new = b + step
                fb_new = eval_f(b_new)
                if fb_new is not None and fa * fb_new < 0:
                    b, fb = b_new, fb_new
                    break
                if fb_new is not None:
                    b, fb = b_new, fb_new
                
                step *= 2
            else:
                return None
        
        # Now we have a bracket [a, b] with fa*fb < 0
        # Set up initial bracket
        a_br, b_br = a, b
        fa_br, fb_br = fa, fb
        
        # Keep track of the best point inside the bracket
        if abs(fa_br) < abs(fb_br):
            x_best_br, f_best_br = a_br, fa_br
        else:
            x_best_br, f_best_br = b_br, fb_br
        
        # Store the third point for quadratic interpolation
        # We need three distinct points for quadratic interpolation
        # Start with a_br, b_br, and the midpoint
        mid_init = (a_br + b_br) / 2
        fm_init = eval_f(mid_init)
        if fm_init is None:
            return None
        
        if abs(fm_init) < abs(f_best_br):
            x_best_br, f_best_br = mid_init, fm_init
        
        # Now we have three points: a_br, b_br, mid_init
        # Use them for quadratic interpolation
        x1, f1 = a_br, fa_br
        x2, f2 = b_br, fb_br
        x3, f3 = mid_init, fm_init
        
        # Main SIQI loop
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            if abs(f_best_br) <= 1e-12:
                return x_best_br
            
            # Try quadratic interpolation using the three most recent points
            # We need to ensure the points are distinct
            use_interp = False
            x_interp = None
            
            # Check if we have three distinct points with finite values
            if (abs(x1 - x2) > 1e-15 * max(1.0, abs(x1)) and 
                abs(x2 - x3) > 1e-15 * max(1.0, abs(x2)) and 
                abs(x1 - x3) > 1e-15 * max(1.0, abs(x1))):
                
                # Compute quadratic interpolation
                # Lagrange form: q(x) = f1*(x-x2)*(x-x3)/((x1-x2)*(x1-x3)) + f2*(x-x1)*(x-x3)/((x2-x1)*(x2-x3)) + f3*(x-x1)*(x-x2)/((x3-x1)*(x3-x2))
                # We want q(x) = 0, which is a quadratic in x
                # Let's solve for x
                denom1 = (x1 - x2) * (x1 - x3)
                denom2 = (x2 - x1) * (x2 - x3)
                denom3 = (x3 - x1) * (x3 - x2)
                
                if abs(denom1) > 1e-300 and abs(denom2) > 1e-300 and abs(denom3) > 1e-300:
                    # Coefficients of the quadratic: Ax^2 + Bx + C = 0
                    A = f1/denom1 + f2/denom2 + f3/denom3
                    B = -f1*(x2+x3)/denom1 - f2*(x1+x3)/denom2 - f3*(x1+x2)/denom3
                    C = f1*x2*x3/denom1 + f2*x1*x3/denom2 + f3*x1*x2/denom3
                    
                    disc = B*B - 4*A*C
                    if disc >= 0 and abs(A) > 1e-300:
                        sqrt_disc = math.sqrt(disc)
                        x_root1 = (-B + sqrt_disc) / (2*A)
                        x_root2 = (-B - sqrt_disc) / (2*A)
                        
                        # Pick the root that's inside the bracket
                        for x_cand in [x_root1, x_root2]:
                            if a_br < x_cand < b_br and math.isfinite(x_cand):
                                x_interp = x_cand
                                use_interp = True
                                break
                        
                        # If no root in bracket, try the one closer to bracket center
                        if not use_interp:
                            mid_br = (a_br + b_br) / 2
                            dist1 = abs(x_root1 - mid_br)
                            dist2 = abs(x_root2 - mid_br)
                            x_cand = x_root1 if dist1 < dist2 else x_root2
                            if math.isfinite(x_cand):
                                x_interp = x_cand
                                use_interp = True
            
            if not use_interp:
                # Fall back to bisection
                x_interp = (a_br + b_br) / 2
            
            fm = eval_f(x_interp)
            if fm is None:
                break
            
            if abs(fm) <= 1e-12:
                return x_interp
            
            # Update bracket
            if fm * fa_br < 0:
                b_br, fb_br = x_interp, fm
                # Update x3 to be the new b_br
                x3, f3 = b_br, fb_br
            else:
                a_br, fa_br = x_interp, fm
                # Update x1 to be the new a_br
                x1, f1 = a_br, fa_br
            
            # Update best
            if abs(fm) < abs(f_best_br):
                x_best_br, f_best_br = x_interp, fm
            
            # Check convergence
            if abs(b_br - a_br) < 1e-15 * max(1.0, abs(x_interp)):
                break
        
        return x_best_br if abs(f_best_br) <= 1e-8 else None
    
    # Try Newton's method with damping
    def try_newton(x_start, fx_start, max_iter=30, tol=1e-8):
        x = x_start
        fx = fx_start
        for i in range(max_iter):
            if time.time() > deadline:
                return None
            if abs(fx) <= tol:
                return x
            
            h = 1e-7 * max(1.0, abs(x))
            fx_plus = eval_f(x + h)
            if fx_plus is None:
                break
            d = (fx_plus - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            
            step = fx / d
            # Damp the step if it's too large
            if abs(step) > abs(x) * 0.5:
                step = math.copysign(min(abs(step), abs(x) * 0.5), step)
            
            x_new = x - step
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            
            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
            else:
                # Try smaller step
                step *= 0.5
                x_new = x - step
                fx_new = eval_f(x_new)
                if fx_new is None:
                    break
                if abs(fx_new) < abs(fx):
                    x, fx = x_new, fx_new
                else:
                    break
        return x if abs(fx) <= tol else None
    
    # Try secant method
    def try_secant(x1, fx1, x2, fx2, max_iter=30):
        for i in range(max_iter):
            if time.time() > deadline:
                return None
            if abs(fx1) <= 1e-12:
                return x1
            if abs(fx2) <= 1e-12:
                return x2
            
            if fx1 == fx2:
                break
            
            x_new = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            
            x1, fx1 = x2, fx2
            x2, fx2 = x_new, fx_new
        
        best = x1 if abs(fx1) <= abs(fx2) else x2
        best_fx = fx1 if abs(fx1) <= abs(fx2) else fx2
        return best if abs(best_fx) <= 1e-8 else None
    
    # Try bisection - need a bracket
    def find_bracket(x_center, fx_center, max_expansions=20):
        """Find a bracket [a, b] where f(a) and f(b) have opposite signs."""
        a, b = x_center, x_center
        fa, fb = fx_center, fx_center
        
        # Expand outward to find sign change
        step_size = max(1.0, abs(x_center)) * 0.1
        for _ in range(max_expansions):
            if time.time() > deadline:
                return None
            
            # Try expanding left
            a_new = a - step_size
            fa_new = eval_f(a_new)
            if fa_new is not None and fa_new * fb < 0:
                return (a_new, b, fa_new, fb)
            if fa_new is not None:
                a, fa = a_new, fa_new
            
            # Try expanding right
            b_new = b + step_size
            fb_new = eval_f(b_new)
            if fb_new is not None and fa * fb_new < 0:
                return (a, b_new, fa, fb_new)
            if fb_new is not None:
                b, fb = b_new, fb_new
            
            step_size *= 2
        
        return None
    
    def try_bisection(a, b, fa, fb, max_iter=50):
        for _ in range(max_iter):
            if time.time() > deadline:
                return None
            mid = (a + b) / 2
            fm = eval_f(mid)
            if fm is None:
                break
            if abs(fm) <= 1e-12:
                return mid
            if fa * fm < 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            if abs(b - a) < 1e-15 * max(1.0, abs(mid)):
                break
        mid = (a + b) / 2
        return mid
    
    # Final high-precision polish pass
    def polish(x_init, fx_init, max_iter=20):
        """
        Final polish using Newton's method with very tight tolerance.
        Uses central difference for derivative for better accuracy.
        """
        x = x_init
        fx = fx_init
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            if abs(fx) <= 1e-14:
                return x
            
            # Use central difference for better derivative estimate
            h = 1e-6 * max(1.0, abs(x))
            fx_plus = eval_f(x + h)
            fx_minus = eval_f(x - h)
            
            if fx_plus is None or fx_minus is None:
                # Fall back to forward difference
                h = 1e-7 * max(1.0, abs(x))
                fx_plus = eval_f(x + h)
                if fx_plus is None:
                    break
                d = (fx_plus - fx) / h
            else:
                d = (fx_plus - fx_minus) / (2 * h)
            
            if d == 0 or not math.isfinite(d):
                break
            
            step = fx / d
            
            # Limit step size
            max_step = 0.1 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = math.copysign(max_step, step)
            
            x_new = x - step
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            
            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
            else:
                # Try halving the step
                for _ in range(5):
                    step *= 0.5
                    x_new = x - step
                    fx_new = eval_f(x_new)
                    if fx_new is None:
                        break
                    if abs(fx_new) < abs(fx):
                        x, fx = x_new, fx_new
                        break
                else:
                    break
        
        return x if abs(fx) <= 1e-8 else None
    
    # First, try SIQI from x0
    result = try_siqi(x, fx)
    if result is not None:
        # Apply final polish
        fx_result = eval_f(result)
        if fx_result is not None:
            polished = polish(result, fx_result, max_iter=10)
            if polished is not None:
                return polished
        return result
    
    # Update best
    if abs(fx) < abs(best_fx):
        best_x, best_fx = x, fx
    
    # Try Newton from x0
    result = try_newton(x, fx)
    if result is not None:
        # Apply final polish
        fx_result = eval_f(result)
        if fx_result is not None:
            polished = polish(result, fx_result, max_iter=10)
            if polished is not None:
                return polished
        return result
    
    # Try secant with a perturbed point
    h = 1e-5 * max(1.0, abs(x))
    x2 = x + h
    fx2 = eval_f(x2)
    if fx2 is not None:
        result = try_secant(x, fx, x2, fx2)
        if result is not None:
            # Apply final polish
            fx_result = eval_f(result)
            if fx_result is not None:
                polished = polish(result, fx_result, max_iter=10)
                if polished is not None:
                    return polished
            return result
    
    # Try to find a bracket and use bisection
    bracket = find_bracket(x, fx)
    if bracket is not None:
        a, b, fa, fb = bracket
        result = try_bisection(a, b, fa, fb)
        if result is not None:
            # Apply final polish
            fx_result = eval_f(result)
            if fx_result is not None:
                polished = polish(result, fx_result, max_iter=10)
                if polished is not None:
                    return polished
            return result
    
    # Multiple restarts with random perturbations
    rng = np.random.RandomState(42)
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        # Random restart around x0 with increasing radius
        radius = 1e-3 * (10 ** attempt) * max(1.0, abs(x0))
        x_try = x0 + rng.normal(0, radius)
        fx_try = eval_f(x_try)
        if fx_try is None:
            continue
        
        if abs(fx_try) < abs(best_fx):
            best_x, best_fx = x_try, fx_try
        
        if abs(fx_try) <= 1e-8:
            # Apply final polish
            polished = polish(x_try, fx_try, max_iter=10)
            if polished is not None:
                return polished
            return x_try
        
        # Try SIQI from this point
        result = try_siqi(x_try, fx_try, max_iter=20)
        if result is not None:
            # Apply final polish
            fx_result = eval_f(result)
            if fx_result is not None:
                polished = polish(result, fx_result, max_iter=10)
                if polished is not None:
                    return polished
            return result
        
        # Try Newton from this point
        result = try_newton(x_try, fx_try, max_iter=15)
        if result is not None:
            # Apply final polish
            fx_result = eval_f(result)
            if fx_result is not None:
                polished = polish(result, fx_result, max_iter=10)
                if polished is not None:
                    return polished
            return result
    
    # Final attempt: polish the best solution found
    if abs(best_fx) <= 1e-4:  # Only polish if we're reasonably close
        polished = polish(best_x, best_fx, max_iter=15)
        if polished is not None:
            return polished
    
    # Return best solution found
    return best_x
# EVOLVE-BLOCK-END
