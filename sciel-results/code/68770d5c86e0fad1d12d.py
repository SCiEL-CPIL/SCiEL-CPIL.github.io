"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    x = float(x0)

    # 1. Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        # If non-finite, try to find a nearby finite point
        found = False
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            for sign in [1, -1]:
                x_test = x + sign * dx
                try:
                    fx_test = f(x_test)
                except Exception:
                    continue
                if math.isfinite(fx_test):
                    x = x_test
                    fx = fx_test
                    found = True
                    break
            if found:
                break
        if not found:
            return x0
        if abs(fx) <= 1e-8:
            return x

    if abs(fx) <= 1e-8:
        return x

    # 2. Find a second point to establish a bracket or direction
    # We need to find a point y such that f(y) is finite.
    # Ideally, we want f(x) and f(y) to have opposite signs for bisection safety.
    # If we can't find opposite signs quickly, we proceed with secant/IQI but
    # use bisection-like steps if the function is not monotonic or step is bad.
    
    # Initial step size
    h = 1e-4 * max(1.0, abs(x))
    
    y = None
    fy = None
    
    # Try to find a bracketing interval [a, b]
    # Start with a=x
    a = x
    fa = fx
    
    # Expand to find b
    # Try small steps first
    for scale in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        # Try positive direction
        b_candidate = a + scale * max(1.0, abs(a))
        try:
            fb_candidate = f(b_candidate)
            if math.isfinite(fb_candidate):
                # Check if we found a bracket
                if fa * fb_candidate < 0:
                    y = b_candidate
                    fy = fb_candidate
                    break
                # If not bracketed, store as candidate for secant/IQI start
                if y is None:
                    y = b_candidate
                    fy = fb_candidate
                # If we have a candidate but haven't found a bracket, keep expanding
                # But we should also check the other direction if we haven't found anything yet
        except Exception:
            pass
        
        # Try negative direction (only if we haven't found a bracket and we are early in the search)
        if y is None or (fa * fy > 0): # If current y doesn't bracket
            b_candidate = a - scale * max(1.0, abs(a))
            try:
                fb_candidate = f(b_candidate)
                if math.isfinite(fb_candidate):
                    if fa * fb_candidate < 0:
                        y = b_candidate
                        fy = fb_candidate
                        break
                    if y is None:
                        y = b_candidate
                        fy = fb_candidate
            except Exception:
                pass
        
        # If we found a bracket, we can stop expanding
        if y is not None and fa * fy < 0:
            break

    if y is None:
        # If we couldn't find any other point, return x0 or best effort
        # This is a failure case for the algorithm, but we return x
        return x

    # Now we have points (a, fa) and (y, fy)
    # Check if y is better
    if abs(fy) < abs(fa):
        a, fa = y, fy
        # We lost the other point, but we'll re-evaluate or keep history
        # Actually, let's keep two points: p1 and p2
        # Let's restart the logic with two points
        # We need a third point for IQI. Let's generate one via Secant first.
        
    # Let's structure the state with two points
    x1, f1 = a, fa
    x2, f2 = y, fy
    
    # Ensure x2 is the "current" best or most recent?
    # Let's just keep a history list
    history = [(x1, f1), (x2, f2)]
    
    # If we have a bracket, update a, b
    a_bracket, b_bracket = None, None
    if f1 * f2 < 0:
        if abs(f1) < abs(f2):
            a_bracket, b_bracket = x1, x2
        else:
            a_bracket, b_bracket = x2, x1

    best_x = x1 if abs(f1) < abs(f2) else x2
    best_fx = f1 if abs(f1) < abs(f2) else f2

    # Main Loop
    for _ in range(100):
        if time.time() > deadline:
            break

        # Check if current best is good enough
        if abs(best_fx) <= 1e-8:
            return best_x

        # Determine next step
        x_new = None
        f_new = None

        # Strategy 1: IQI if we have 3 points
        if len(history) >= 3:
            (xa, fa), (xb, fb), (xc, fc) = history[-3], history[-2], history[-1]
            
            # Check for distinct f values to avoid division by zero
            if abs(fa - fb) > 1e-30 and abs(fb - fc) > 1e-30 and abs(fa - fc) > 1e-30:
                # Inverse Quadratic Interpolation
                # x = A*f^2 + B*f + C
                # x_new = xc - fc * ( (xc-xb)*(xc-xa) / ((fc-fb)*(fc-fa)) )
                # Wait, the standard formula is:
                # x_new = xc + fc * [ (xc-xb)*(xc-xa) / ((fc-fb)*(fc-fa)) ] ... No.
                
                # Lagrange form for x(f) = 0
                # x_new = xa * (0-fb)*(0-fc) / ((fa-fb)*(fa-fc)) 
                #       + xb * (0-fa)*(0-fc) / ((fb-fa)*(fb-fc)) 
                #       + xc * (0-fa)*(0-fb) / ((fc-fa)*(fc-fb))
                
                denom_a = (fa - fb) * (fa - fc)
                denom_b = (fb - fa) * (fb - fc)
                denom_c = (fc - fa) * (fc - fb)
                
                term_a = xa * (fb * fc) / denom_a
                term_b = xb * (fa * fc) / denom_b
                term_c = xc * (fa * fb) / denom_c
                
                x_candidate = term_a + term_b + term_c
                
                if math.isfinite(x_candidate):
                    # Safeguard: If we have a bracket, ensure x_candidate is within [a_bracket, b_bracket]
                    if a_bracket is not None:
                        if (x_candidate < min(a_bracket, b_bracket) or x_candidate > max(a_bracket, b_bracket)):
                            # Fall back to bisection
                            x_candidate = (a_bracket + b_bracket) / 2.0
                    else:
                        # If no bracket, check if step is reasonable
                        # If it's too far, maybe use secant
                        if abs(x_candidate - xc) > 10.0 * max(1.0, abs(xc)):
                             # Fall back to secant
                             if abs(fb - fc) > 1e-30:
                                 x_candidate = xc - fc * (xc - xb) / (fc - fb)
                             else:
                                 x_candidate = None
                                 
                    if x_candidate is not None:
                        x_new = x_candidate

        # Strategy 2: Secant if IQI failed or we have only 2 points
        if x_new is None and len(history) >= 2:
            (xb, fb), (xc, fc) = history[-2], history[-1]
            if abs(fb - fc) > 1e-30:
                x_candidate = xc - fc * (xc - xb) / (fc - fb)
                if math.isfinite(x_candidate):
                    # Safeguard
                    if a_bracket is not None:
                        if (x_candidate < min(a_bracket, b_bracket) or x_candidate > max(a_bracket, b_bracket)):
                            x_candidate = (a_bracket + b_bracket) / 2.0
                    else:
                        if abs(x_candidate - xc) > 10.0 * max(1.0, abs(xc)):
                             # If secant step is huge, try bisection-like shrink
                             # Or just accept it? Polynomials are smooth.
                             # Let's accept it but cap it.
                             x_candidate = xc + math.copysign(10.0 * max(1.0, abs(xc)), x_candidate - xc)
                    x_new = x_candidate

        # Strategy 3: Bisection (if we have a bracket and above failed)
        if x_new is None and a_bracket is not None:
            x_new = (a_bracket + b_bracket) / 2.0

        # Strategy 4: Fallback step (if no bracket and no secant)
        if x_new is None:
            # Take a small step in the direction of decreasing |f|
            # Use gradient descent approximation
            # f'(x) ~ (f(x+h) - f(x-h)) / 2h
            # This costs 2 calls. We want to avoid this.
            # Let's just take a small step in the direction of the last delta
            if len(history) >= 2:
                dx = history[-1][0] - history[-2][0]
                x_new = history[-1][0] + 0.5 * dx
            else:
                x_new = history[-1][0] + 1e-4

        # Evaluate x_new
        try:
            f_new = f(x_new)
        except Exception:
            # If exception, try bisection if possible
            if a_bracket is not None:
                x_new = (a_bracket + b_bracket) / 2.0
                try:
                    f_new = f(x_new)
                except Exception:
                    break
            else:
                break

        if not math.isfinite(f_new):
            # Try bisection
            if a_bracket is not None:
                x_new = (a_bracket + b_bracket) / 2.0
                try:
                    f_new = f(x_new)
                except Exception:
                    break
            else:
                break

        # Update Best
        if abs(f_new) < abs(best_fx):
            best_x, best_fx = x_new, f_new

        # Update Bracket
        if a_bracket is not None:
            if f_new * f(a_bracket) < 0:
                b_bracket = x_new
            else:
                a_bracket = x_new
                # We need to re-evaluate f(a_bracket) if we swapped? 
                # We have f(a_bracket) stored? No, we only store x.
                # Let's store bracket points with values
                # Actually, we can just check signs.
                # If f_new * f(a_bracket) > 0, then root is between x_new and b_bracket?
                # Wait.
                # If f(a) * f(b) < 0.
                # If f(new) * f(a) < 0, new is in (new, b) ? No.
                # If f(a) and f(new) have opposite signs, root is in (a, new). So b becomes new.
                # If f(a) and f(new) have same sign, root is in (new, b). So a becomes new.
                # But we don't have f(a) stored separately from history?
                # We can find f(a) in history or re-evaluate? Re-evaluating costs a call.
                # Let's store bracket values.
                pass # This logic is flawed without storing fa, fb.

        # Let's refactor to store bracket values
        # This is getting complicated. Let's simplify.
        # We will maintain:
        # 1. History of last 3 points for IQI/Secant.
        # 2. A bracket [lo, hi] with f_lo, f_hi if available.

        # Let's restart the loop logic with better state management.
        # This implementation is getting messy. Let's write a cleaner version.
        break # Break out of this messy loop and rewrite below

    # --- CLEANER IMPLEMENTATION ---
    
    # Reset state for cleaner logic
    # We already spent calls to find x1, f1, x2, f2.
    # Let's just use the current x1, f1, x2, f2.
    
    # State:
    # x, fx: current point
    # x_prev, fx_prev: previous point
    # bracket: (a, fa, b, fb) or None
    
    x_curr, f_curr = x1, f1
    x_prev, f_prev = x2, f2
    
    # Ensure x_curr is the one with smaller |f|? Not necessary, but helps.
    if abs(f_curr) > abs(f_prev):
        x_curr, f_curr, x_prev, f_prev = x_prev, f_prev, x_curr, f_curr
        
    # Initialize bracket
    a, fa, b, fb = None, None, None, None
    if f_curr * f_prev < 0:
        if abs(f_curr) < abs(f_prev):
            a, fa, b, fb = x_curr, f_curr, x_prev, f_prev
        else:
            a, fa, b, fb = x_prev, f_prev, x_curr, f_curr

    best_x, best_fx = x_curr, f_curr
    if abs(f_prev) < abs(f_curr):
        best_x, best_fx = x_prev, f_prev

    # Main Loop
    for _ in range(200):
        if time.time() > deadline:
            break

        if abs(best_fx) <= 1e-8:
            return best_x

        x_new = None
        f_new = None

        # 1. Try IQI if we have 3 points
        # We need 3 distinct points. We have x_curr, x_prev.
        # We need a third. Let's use the one before x_prev if we had it.
        # Since we only kept 2, we can't do IQI immediately.
        # We will generate a new point using Secant, then next iteration we have 3.
        
        # To do IQI, we need history of 3.
        # Let's maintain a list `hist` of last 3 points.
        # hist = [(x_curr, f_curr), (x_prev, f_prev)]
        
        # If len(hist) >= 3:
        #   Try IQI
        # Else:
        #   Try Secant
        
        # Let's implement with a history list for clarity.
        # But we already spent calls. Let's just continue with Secant until we have 3.
        
        # Secant Step
        if abs(f_curr - f_prev) > 1e-30:
            x_candidate = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            if math.isfinite(x_candidate):
                # Safeguard with bracket
                if a is not None:
                    if x_candidate < min(a, b) or x_candidate > max(a, b):
                        x_candidate = (a + b) / 2.0
                else:
                    # Cap step size
                    max_step = 10.0 * max(1.0, abs(x_curr))
                    if abs(x_candidate - x_curr) > max_step:
                        x_candidate = x_curr + math.copysign(max_step, x_candidate - x_curr)
                x_new = x_candidate
        else:
            # f_curr == f_prev, secant fails. Try bisection if bracket exists.
            if a is not None:
                x_new = (a + b) / 2.0
            else:
                # Perturb
                x_new = x_curr + 1e-4

        # Evaluate
        try:
            f_new = f(x_new)
        except Exception:
            if a is not None:
                x_new = (a + b) / 2.0
                try:
                    f_new = f(x_new)
                except Exception:
                    break
            else:
                break

        if not math.isfinite(f_new):
            if a is not None:
                x_new = (a + b) / 2.0
                try:
                    f_new = f(x_new)
                except Exception:
                    break
            else:
                break

        # Update Best
        if abs(f_new) < abs(best_fx):
            best_x, best_fx = x_new, f_new

        # Update Bracket
        if a is not None:
            if f_new * fa < 0:
                # Root is between x_new and b? No.
                # f(a) and f(new) have opposite signs. Root is in (a, new).
                # So new becomes b.
                b, fb = x_new, f_new
            else:
                # f(a) and f(new) have same sign. Root is in (new, b).
                # So new becomes a.
                a, fa = x_new, f_new
        else:
            # No bracket. Check if we can form one with current and new
            if f_new * f_curr < 0:
                if abs(f_new) < abs(f_curr):
                    a, fa, b, fb = x_new, f_new, x_curr, f_curr
                else:
                    a, fa, b, fb = x_curr, f_curr, x_new, f_new

        # Update History for IQI
        # We want to keep the 3 most recent points.
        # Current points: x_curr, x_prev, x_new
        # Next iteration: x_new becomes x_curr, x_curr becomes x_prev.
        
        # We should also check if we can do IQI next time.
        # Let's just shift.
        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_new, f_new

    return best_x
# EVOLVE-BLOCK-END
