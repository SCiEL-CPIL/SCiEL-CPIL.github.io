"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using a hybrid safeguarded Newton/secant
    method with bisection fallback and strict call/time budgets.
    """
    start_time = time.time()
    # Use a conservative deadline to avoid the 150s kill; leave margin for evaluator overhead
    DEADLINE = start_time + 130.0
    MAX_CALLS = 480  # Stay well under 500
    TOL = 1e-8
    
    call_count = 0
    best_x = float(x0)
    best_fval = float('inf')
    
    def safe_call(x):
        """Call f(x) with budget and finiteness checks. Returns (value, success_flag)."""
        nonlocal call_count, best_x, best_fval
        if call_count >= MAX_CALLS:
            return None, False
        if time.time() > DEADLINE:
            return None, False
        
        call_count += 1
        try:
            val = f(x)
            if not isinstance(val, (int, float)):
                return None, False
            val = float(val)
            if not math.isfinite(val):
                return None, False
            if abs(val) < best_fval:
                best_fval = abs(val)
                best_x = x
            return val, True
        except Exception:
            return None, False

    x = float(x0)
    
    # Initial evaluation
    fx, ok = safe_call(x)
    if not ok:
        return best_x
    if abs(fx) <= TOL:
        return x
    
    # Try to find a bracketing interval [a, b]
    a, b = None, None
    fa, fb = None, None
    
    # Start with a small step and expand
    step = max(1e-4, abs(x) * 1e-4)
    # Try a few steps to find a sign change
    # We already have f(x). Let's try x + step and x - step.
    # But we want to be efficient.
    
    # Strategy:
    # 1. Try to find a bracket.
    # 2. If found, use Bisection/Newton hybrid.
    # 3. If not, use Safeguarded Secant.
    
    bracket_found = False
    
    # Try expanding interval from x
    # We have fx at x.
    # Try x + step
    x_plus = x + step
    f_plus, ok = safe_call(x_plus)
    if ok:
        if abs(f_plus) <= TOL:
            return x_plus
        if fx * f_plus < 0:
            a, b = x, x_plus
            fa, fb = fx, f_plus
            bracket_found = True
        else:
            # Try x - step
            x_minus = x - step
            f_minus, ok = safe_call(x_minus)
            if ok:
                if abs(f_minus) <= TOL:
                    return x_minus
                if fx * f_minus < 0:
                    a, b = x_minus, x
                    fa, fb = f_minus, fx
                    bracket_found = True
                else:
                    # Expand further
                    # Keep expanding until we find a sign change or exhaust budget
                    # We have points: x_minus, x, x_plus
                    # Let's expand the side that had the same sign as fx?
                    # Actually, just expand the larger interval.
                    # Current interval candidates: [x_minus, x_plus]
                    # If f(x_minus) and f(x_plus) have opposite signs, we are done.
                    if f_minus * f_plus < 0:
                        a, b = x_minus, x_plus
                        fa, fb = f_minus, f_plus
                        bracket_found = True
                    else:
                        # Need to expand.
                        # Determine direction.
                        # If f(x) and f(x_plus) have same sign, expand to right.
                        # If f(x) and f(x_minus) have same sign, expand to left.
                        # If both same, pick one.
                        
                        # Let's just do a simple expansion loop
                        # We have left point L, right point R.
                        # Initialize L=x_minus, R=x_plus
                        L = x_minus
                        R = x_plus
                        fL = f_minus
                        fR = f_plus
                        
                        # Ensure fL and fR are defined
                        if fL is None or fR is None:
                            bracket_found = False
                        else:
                            # Expand
                            for _ in range(20):
                                if not bracket_found and call_count < MAX_CALLS and time.time() < DEADLINE:
                                    if fL * fR < 0:
                                        a, b = L, R
                                        fa, fb = fL, fR
                                        bracket_found = True
                                        break
                                    
                                    # Expand the side where sign is same as the middle?
                                    # Or just double the interval.
                                    # Let's expand the side that is "further" from the root?
                                    # Heuristic: if |fL| < |fR|, expand R. Else expand L.
                                    # Or just alternate.
                                    
                                    # Let's try expanding the side with larger |f|
                                    if abs(fL) > abs(fR):
                                        # Expand R
                                        new_R = R + (R - L)
                                        f_new_R, ok = safe_call(new_R)
                                        if not ok:
                                            break
                                        if abs(f_new_R) <= TOL:
                                            return new_R
                                        R = new_R
                                        fR = f_new_R
                                    else:
                                        # Expand L
                                        new_L = L - (R - L)
                                        f_new_L, ok = safe_call(new_L)
                                        if not ok:
                                            break
                                        if abs(f_new_L) <= TOL:
                                            return new_L
                                        L = new_L
                                        fL = f_new_L
                                else:
                                    break
                                # Check if we found bracket
                                if fL * fR < 0:
                                    a, b = L, R
                                    fa, fb = fL, fR
                                    bracket_found = True
                                    break
                                # Check bounds
                                if abs(R - L) > 1e10:
                                    break

    if bracket_found:
        # Use Bisection with Newton acceleration (Brent's method style but simpler)
        # We have [a, b] with fa, fb of opposite signs.
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        for _ in range(40):
            if call_count >= MAX_CALLS or time.time() > DEADLINE:
                break
            
            # Check endpoints
            if abs(fa) <= TOL:
                return a
            if abs(fb) <= TOL:
                return b
            
            # Midpoint
            m = (a + b) / 2.0
            fm, ok = safe_call(m)
            if not ok:
                break
            if abs(fm) <= TOL:
                return m
            
            # Update bracket
            if fa * fm < 0:
                b = m
                fb = fm
            else:
                a = m
                fa = fm
            
            # Optional: Newton step if derivative is available (approximate)
            # For now, stick to bisection for robustness, or add a secant-like step?
            # Bisection is guaranteed to converge.
            # To speed up, we can try a secant step between a and b.
            # But we need to keep it within [a,b].
            
            # Let's try a secant step
            if fa != fb:
                x_secant = b - fb * (b - a) / (fb - fa)
                # Clamp to [a, b]
                if a < x_secant < b:
                    # Try it
                    # But we already called f(m).
                    # We can just return if it's good enough.
                    # For simplicity and safety, stick to bisection if we don't want to risk it.
                    # However, bisection is slow (log2(500) ~ 9 iterations to get 1e-3, ~27 for 1e-8).
                    # 27 calls is fine.
                    pass
    
    # If no bracket found, use Safeguarded Secant
    # We have x, fx.
    # We need another point.
    # Use x + h
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    fx1, ok = safe_call(x1)
    
    if not ok:
        return best_x
    
    x0_curr = x
    fx0_curr = fx
    x1_curr = x1
    fx1_curr = fx1
    
    for _ in range(40):
        if call_count >= MAX_CALLS or time.time() > DEADLINE:
            break
        
        if abs(fx0_curr) <= TOL:
            return x0_curr
        if abs(fx1_curr) <= TOL:
            return x1_curr
        
        if fx0_curr == fx1_curr:
            # Degenerate, perturb
            x1_curr = x0_curr + h * 0.5
            fx1_curr, ok = safe_call(x1_curr)
            if not ok:
                break
            continue
        
        # Secant step
        denom = fx1_curr - fx0_curr
        x_new = x1_curr - fx1_curr * (x1_curr - x0_curr) / denom
        
        if not math.isfinite(x_new):
            break
        
        # Safeguard: keep x_new within a reasonable range of x1_curr
        # If it's too far, use bisection-like step or damping
        max_step = 10.0 * max(1.0, abs(x1_curr))
        if abs(x_new - x1_curr) > max_step:
            # Damp
            x_new = x1_curr + max_step * math.copysign(1.0, x_new - x1_curr)
        
        fx_new, ok = safe_call(x_new)
        if not ok:
            break
        
        # Update
        x0_curr = x1_curr
        fx0_curr = fx1_curr
        x1_curr = x_new
        fx1_curr = fx_new
    
    # Return the best found
    return best_x
# EVOLVE-BLOCK-END
