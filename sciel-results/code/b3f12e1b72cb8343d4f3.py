"""Root finding for ill-conditioned equations: Global Bracket Search with Illinois Polish."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin below 120s limit
    MAX_EVALS = 480   # Safety margin below 500 limit
    
    eval_count = 0
    best_x = float(x0)
    best_f = None
    best_abs_f = float('inf')

    def safe_eval(x):
        nonlocal eval_count, best_x, best_f, best_abs_f
        if eval_count >= MAX_EVALS or (time.time() - start_time) > DEADLINE:
            return None
        try:
            val = f(x)
            if not math.isfinite(val):
                return None
            eval_count += 1
            abs_val = abs(val)
            if abs_val < best_abs_f:
                best_f = val
                best_x = x
                best_abs_f = abs_val
            return val
        except Exception:
            return None

    def check_budget():
        return eval_count >= MAX_EVALS or (time.time() - start_time) > DEADLINE

    # 1. Initial Check
    f0 = safe_eval(best_x)
    if f0 is not None and abs(f0) <= 1e-8:
        return best_x

    # 2. Global Bracket Search
    # We search for a sign change [a, b] by expanding radially from x0.
    # This is robust against ill-conditioning because it doesn't rely on local derivatives.
    # We use a logarithmic-ish expansion to cover a wide range efficiently.
    
    a, b, fa, fb = None, None, None, None
    
    # Initial points
    x_curr = float(x0)
    f_curr = f0
    
    if f_curr is None:
        # If f(x0) failed, try a tiny step
        x_curr = x0 + 1e-4
        f_curr = safe_eval(x_curr)
        if f_curr is None:
            return best_x

    # Expansion strategy:
    # Try points at distances: 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000
    # in both positive and negative directions.
    # We check for sign change between consecutive points or between current and new.
    
    distances = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]
    
    # Keep track of the last valid point in each direction to form brackets
    last_pos_x, last_pos_f = x_curr, f_curr
    last_neg_x, last_neg_f = x_curr, f_curr
    
    found_bracket = False
    bracket_a, bracket_b, bracket_fa, bracket_fb = None, None, None, None

    # We will iterate through distances.
    # For each distance d, we check x0 + d and x0 - d.
    # If a sign change is found between the previous point and the new point, we have a bracket.
    
    for d in distances:
        if check_budget():
            break
            
        # Check positive direction
        x_plus = x0 + d
        f_plus = safe_eval(x_plus)
        if f_plus is not None:
            if last_pos_f is not None:
                if last_pos_f * f_plus < 0:
                    bracket_a, bracket_b = last_pos_x, x_plus
                    bracket_fa, bracket_fb = last_pos_f, f_plus
                    found_bracket = True
                    break
            # Update last point if we didn't find a bracket yet, 
            # but only if the function value is finite (which it is)
            # We keep the closest point to x0 as the 'last' for bracketing 
            # against the next further point? 
            # Actually, for bracketing, we want to check the interval [prev, curr].
            # So we should update last_pos to the current x_plus if no bracket found?
            # Yes, because if sign change is between d1 and d2, we need both.
            last_pos_x, last_pos_f = x_plus, f_plus
            
        # Check negative direction
        x_minus = x0 - d
        f_minus = safe_eval(x_minus)
        if f_minus is not None:
            if last_neg_f is not None:
                if last_neg_f * f_minus < 0:
                    bracket_a, bracket_b = x_minus, last_neg_x
                    bracket_fa, bracket_fb = f_minus, last_neg_f
                    found_bracket = True
                    break
            last_neg_x, last_neg_f = x_minus, f_minus
            
        # If we've used too many evals, stop searching
        if eval_count > 200:
            break

    if found_bracket:
        # 3. Illinois Method Polish
        # Illinois is a variant of Regula Falsi that damps the side of the bracket
        # that stays the same, accelerating convergence compared to standard Regula Falsi
        # while retaining the guaranteed convergence of bisection.
        
        a, b, fa, fb = bracket_a, bracket_b, bracket_fa, bracket_fb
        
        # Ensure |fa| < |fb| for standard Illinois implementation logic
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
            
        # Iterate until convergence or budget
        for _ in range(200):
            if check_budget():
                break
                
            # Illinois step
            if fa * fb > 0:
                # Should not happen if bracket is valid, but safety check
                break
                
            # Calculate new point
            # x_new = (a*fb - b*fa) / (fb - fa)
            denom = fb - fa
            if denom == 0:
                break
            c = (a * fb - b * fa) / denom
            fc = safe_eval(c)
            
            if fc is None:
                break
                
            if abs(fc) <= 1e-8:
                return c
                
            # Update bracket
            if fa * fc < 0:
                # Root is in [a, c]
                b, fb = c, fc
            else:
                # Root is in [c, b]
                # Apply damping to the side that was NOT updated (a)
                # Standard Illinois: multiply the function value of the unchanged endpoint by 0.5
                a, fa = c, fc
                # Wait, standard Illinois updates the endpoint that did NOT change.
                # If we update [c, b], then 'a' is the one that changed to c.
                # The one that stayed is 'b'.
                # Let's re-verify Illinois logic.
                # If f(a)*f(c) > 0, root is in [c, b].
                # New a = c, new fa = fc.
                # Old a is discarded.
                # The damping is applied to the function value of the point that remains.
                # In [c, b], 'b' remains. So we should damp fb?
                # No, the standard algorithm damps the function value of the point 
                # that is NOT the new point? 
                # Let's look at a reference.
                # "If f(a)f(c) > 0, set a=c, f(a)=f(c), and f(b)=f(b)/2"
                # Yes, damp the one that stayed (b).
                # So if we update a to c, we damp b.
                # If we update b to c, we damp a.
                
                # My logic above:
                # if fa * fc < 0: root in [a, c]. b becomes c. a stays. Damp a.
                # else: root in [c, b]. a becomes c. b stays. Damp b.
                
                # Let's re-evaluate the branch.
                # I wrote:
                # if fa * fc < 0: b, fb = c, fc (b changes, a stays) -> Damp fa
                # else: a, fa = c, fc (a changes, b stays) -> Damp fb
                
                # So in the 'else' block (root in [c, b]), we should damp fb.
                fb *= 0.5
            # Wait, I need to implement the damping in the correct branch.
            # Let's rewrite the update logic clearly.
            
            # Re-do update logic for clarity
            if fa * fc < 0:
                # Root in [a, c]
                b, fb = c, fc
                # a stays, so damp fa
                fa *= 0.5
            else:
                # Root in [c, b]
                a, fa = c, fc
                # b stays, so damp fb
                fb *= 0.5
                
            # Convergence check on interval size
            if abs(b - a) < 1e-14:
                return (a + b) / 2.0

        # Return the best point in the final bracket
        return best_x

    else:
        # No bracket found. 
        # Fall back to a local search or return best_x.
        # Given the "ill-conditioned" nature, if we can't bracket, 
        # maybe the root is very close to x0 and we just need to polish 
        # using a method that doesn't require a bracket?
        # Or maybe the function doesn't change sign (tangent root).
        # For tangent roots, f(x) >= 0 or f(x) <= 0 everywhere.
        # In that case, minimizing f(x)^2 is the goal.
        
        # Simple local minimization of f(x)^2 using a few secant-like steps
        # on the residual squared?
        # Or just return best_x.
        
        # Let's try a few secant steps on f(x) from best_x
        x_curr = best_x
        f_curr = best_f
        
        if f_curr is None:
             return best_x

        h = 1e-4 * max(1.0, abs(x_curr))
        x_prev = x_curr + h
        f_prev = safe_eval(x_prev)
        
        if f_prev is None:
             return best_x

        for _ in range(50):
            if check_budget():
                break
            if f_curr == f_prev:
                break
            x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            
            # Safeguard
            max_jump = max(1.0, 0.5 * abs(x_curr))
            if abs(x_new - x_curr) > max_jump:
                x_new = x_curr + math.copysign(max_jump, x_new - x_curr)
                
            f_new = safe_eval(x_new)
            if f_new is None:
                break
                
            if abs(f_new) <= 1e-8:
                return x_new
                
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_new, f_new
            
            if abs(x_new - x_prev) < 1e-14:
                break
                
        return best_x

# EVOLVE-BLOCK-END
