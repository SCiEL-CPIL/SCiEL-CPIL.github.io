import math
import time
import numpy as np

def find_root(f, x0):
    start_time = time.time()
    # Safety margin: 140s out of 150s limit
    deadline = start_time + 140.0
    
    def timed_out():
        return time.time() > deadline

    x = float(x0)
    fx = f(x)
    eval_count = 1
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)
    
    if best_abs_fx <= 1e-8:
        return best_x

    # Phase 1: Bracket Finding
    # We want to find a, b such that f(a) * f(b) < 0.
    # Strategy:
    # 1. Check immediate neighbors.
    # 2. Expand adaptively.
    # 3. If no sign change found, try to minimize |f(x)| locally (quasi-Newton on |f|^2 or just golden section on |f|).
    
    a = x
    b = x
    fa = fx
    fb = fx
    
    bracket_found = False
    
    # Small initial steps to check for sign change near x0
    # Use relative steps
    h = 1e-4 * max(1.0, abs(x))
    
    # Try a few small offsets first to catch nearby roots
    candidates = [x - h, x + h, x - 10*h, x + 10*h]
    
    for cx in candidates:
        if timed_out(): break
        cf = f(cx)
        eval_count += 1
        if math.isnan(cf) or math.isinf(cf):
            continue
        if abs(cf) < best_abs_fx:
            best_x = cx
            best_fx = cf
            best_abs_fx = abs(cf)
        
        if best_abs_fx <= 1e-8:
            return best_x
            
        # Check sign change with current x (or best_x? usually x0 is the anchor)
        # We maintain a bracket relative to the starting point x or the current best?
        # Standard approach: expand from x0.
        if cf * fx < 0:
            if fx < cf: # just for consistent ordering, not strictly needed
                a, b = x, cx
                fa, fb = fx, cf
            else:
                a, b = cx, x
                fa, fb = cf, fx
            bracket_found = True
            break
            
    # If no bracket found in small steps, expand exponentially
    if not bracket_found:
        step = h
        # Expand left and right
        # We keep track of the point with smallest |f| to fall back to minimization if needed
        left_x = x - step
        right_x = x + step
        
        # Evaluate initial expansion
        fl = f(left_x)
        fr = f(right_x)
        eval_count += 2
        
        if timed_out():
            return best_x
            
        if not (math.isnan(fl) or math.isinf(fl)) and abs(fl) < best_abs_fx:
            best_x = left_x
            best_fx = fl
            best_abs_fx = abs(fl)
        if not (math.isnan(fr) or math.isinf(fr)) and abs(fr) < best_abs_fx:
            best_x = right_x
            best_fx = fr
            best_abs_fx = abs(fr)
            
        if best_abs_fx <= 1e-8:
            return best_x
            
        # Check bracket
        if not (math.isnan(fl) or math.isinf(fl)) and fl * fx < 0:
            a, b = left_x, x
            fa, fb = fl, fx
            bracket_found = True
        elif not (math.isnan(fr) or math.isinf(fr)) and fr * fx < 0:
            a, b = x, right_x
            fa, fb = fx, fr
            bracket_found = True
            
        # Expand if no bracket
        expand_count = 0
        max_expand = 50
        while not bracket_found and expand_count < max_expand:
            if timed_out(): break
            
            step *= 2.0
            if step > 1e6 * max(1.0, abs(x)):
                break
                
            new_left = x - step
            new_right = x + step
            
            fl = f(new_left)
            fr = f(new_right)
            eval_count += 2
            expand_count += 1
            
            if timed_out():
                break
                
            if not (math.isnan(fl) or math.isinf(fl)) and abs(fl) < best_abs_fx:
                best_x = new_left
                best_fx = fl
                best_abs_fx = abs(fl)
            if not (math.isnan(fr) or math.isinf(fr)) and abs(fr) < best_abs_fx:
                best_x = new_right
                best_fx = fr
                best_abs_fx = abs(fr)
                
            if best_abs_fx <= 1e-8:
                return best_x
                
            # Check bracket with the previous points?
            # We have points: x, left_x, right_x, new_left, new_right
            # Check if sign changed between new_left and x
            if not (math.isnan(fl) or math.isinf(fl)) and fl * fx < 0:
                a, b = new_left, x
                fa, fb = fl, fx
                bracket_found = True
            elif not (math.isnan(fr) or math.isinf(fr)) and fr * fx < 0:
                a, b = x, new_right
                fa, fb = fx, fr
                bracket_found = True
            
            # Also check if sign changed between new_left and new_right (if x is not between them? 
            # No, x is between new_left and new_right.
            # If fl and fr have opposite signs, there is a root in [new_left, new_right].
            # But we don't necessarily have a point inside with a sign change relative to the ends 
            # unless we check against x.
            # Actually, if fl * fr < 0, we have a bracket [new_left, new_right].
            if not (math.isnan(fl) or math.isinf(fl)) and not (math.isnan(fr) or math.isinf(fr)):
                if fl * fr < 0:
                    a, b = new_left, new_right
                    fa, fb = fl, fr
                    bracket_found = True
                    break

    if bracket_found:
        # Phase 2: Root Refinement
        # Use a robust method: Illinois Method (modified Bisection/Secant)
        # Or Brent's Method. Brent's is faster but complex. Illinois is very robust.
        # Given "ill-conditioned", robustness is key.
        # Let's implement a simple, robust Illinois-like step.
        
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        # We maintain a bracket [a, b] with fa * fb < 0.
        # We want to minimize |f(x)|.
        
        # Illinois Algorithm:
        # 1. Let x1, x2 be endpoints.
        # 2. Compute x3 using secant formula.
        # 3. If x3 is outside [x1, x2] or |f(x3)| > |f(x_prev_best)|, use bisection.
        # 4. Replace the endpoint with larger |f| with x3.
        
        x1, x2 = a, b
        f1, f2 = fa, fb
        
        # Keep track of the best point found during refinement
        # best_x, best_fx, best_abs_fx are already tracked globally
        
        # We will iterate up to a limit or until convergence
        # 500 evals total. We used ~2-10 for bracketing.
        # Let's allow 400 refinement steps.
        
        # To speed up convergence, we can use a hybrid:
        # Try Secant/Inverse Quadratic Interpolation (IQI) if safe, else Bisection.
        
        # State for interpolation:
        # We need at least 3 points for IQI.
        # Let's start with Secant (2 points) and upgrade if we have 3.
        
        # Actually, for ill-conditioned problems, Bisection is the safest.
        # Illinois is Bisection with a Secant attempt.
        
        prev_x = None
        prev_f = None
        
        for _ in range(450):
            if timed_out():
                break
                
            # Check convergence
            if abs(f1) <= 1e-8:
                best_x = x1
                best_fx = f1
                best_abs_fx = abs(f1)
                return best_x
            if abs(f2) <= 1e-8:
                best_x = x2
                best_fx = f2
                best_abs_fx = abs(f2)
                return best_x
                
            # Determine which end to replace
            # We want to replace the end with larger |f|
            if abs(f1) > abs(f2):
                x_bad, f_bad = x1, f1
                x_good, f_good = x2, f2
            else:
                x_bad, f_bad = x2, f2
                x_good, f_good = x1, f1
            
            # Calculate next point
            # Secant step
            denom = f_good - f_bad
            if abs(denom) < 1e-15:
                x_new = (x_bad + x_good) / 2.0
            else:
                x_new = x_bad - f_bad * (x_good - x_bad) / denom
            
            # Check if x_new is within bracket
            # (x_new - x_bad) * (x_new - x_good) > 0 means outside
            if (x_new - x_bad) * (x_new - x_good) > 0:
                x_new = (x_bad + x_good) / 2.0
            
            # Evaluate
            f_new = f(x_new)
            eval_count += 1
            if timed_out():
                break
                
            if math.isnan(f_new) or math.isinf(f_new):
                # Fallback to bisection if evaluation fails
                x_new = (x_bad + x_good) / 2.0
                f_new = f(x_new)
                eval_count += 1
                if timed_out():
                    break
                if math.isnan(f_new) or math.isinf(f_new):
                    break

            # Update best
            if abs(f_new) < best_abs_fx:
                best_x = x_new
                best_fx = f_new
                best_abs_fx = abs(f_new)
                
            if best_abs_fx <= 1e-8:
                return best_x
                
            # Update bracket
            # If f_new has same sign as f_bad, replace x_bad
            # If f_new has same sign as f_good, replace x_good
            # If signs are opposite to both (shouldn't happen in continuous f), pick smaller |f| replacement?
            
            if f_new * f_bad > 0:
                # Same sign as bad, so root is between x_new and x_good
                # Replace x_bad with x_new
                if x_bad == x1:
                    x1 = x_new
                    f1 = f_new
                else:
                    x2 = x_new
                    f2 = f_new
            elif f_new * f_good > 0:
                # Same sign as good, so root is between x_bad and x_new
                # Replace x_good with x_new
                if x_good == x1:
                    x1 = x_new
                    f1 = f_new
                else:
                    x2 = x_new
                    f2 = f_new
            else:
                # f_new is 0 or sign is weird (e.g. -0.0)
                # If f_new is 0, we found root
                if abs(f_new) <= 1e-8:
                    return x_new
                # If not, just pick the side with smaller |f| to replace?
                # Actually, if f_new * f_bad < 0 AND f_new * f_good < 0, it means f_new is 0 (or very close).
                # Or numerical noise.
                # Let's just replace the one with larger |f|
                if abs(f_bad) > abs(f_good):
                     if x_bad == x1:
                         x1 = x_new
                         f1 = f_new
                     else:
                         x2 = x_new
                         f2 = f_new
                else:
                     if x_good == x1:
                         x1 = x_new
                         f1 = f_new
                     else:
                         x2 = x_new
                         f2 = f_new

            # Convergence check based on interval width
            if abs(x1 - x2) < 1e-15 * max(1.0, abs(x1)):
                # Return the one with smaller |f|
                if abs(f1) < abs(f2):
                    return x1
                else:
                    return x2

    # If no bracket was found, we are in trouble.
    # We have tried to expand. If we still don't have a bracket,
    # we might be on a function that doesn't cross zero (e.g. (x-1)^2 + 1e-10).
    # In this case, we should return the point with the minimum |f| found.
    
    return best_x

