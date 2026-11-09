"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    
    Strategy:
    1. Global Adaptive Bracketing: Expand a window around x0 to find a sign change.
       This is robust for polynomials which are continuous and smooth.
    2. Illinois/Regula Falsi Hybrid: Once bracketed, use a method that combines
       the linear convergence of Regula Falsi with the quadratic convergence of
       Bisection to minimize function calls.
    3. Fallback: If no bracket is found (e.g., even multiplicity root or bad x0),
       use Nelder-Mead style local minimization of |f(x)| or a robust Secant method.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    max_calls = 450  # Stay well under 500 to allow for overhead
    
    call_count = 0
    best_x = float(x0)
    best_f = None
    best_abs_f = float('inf')

    def _f(x):
        nonlocal call_count, best_x, best_f, best_abs_f
        if call_count >= max_calls:
            raise RuntimeError("call limit")
        if time.time() > deadline:
            raise RuntimeError("time limit")
        
        call_count += 1
        try:
            val = f(x)
        except Exception:
            val = float('inf')
            
        if math.isfinite(val):
            abs_val = abs(val)
            if abs_val < best_abs_f:
                best_abs_f = abs_val
                best_x = x
                best_f = val
            return val
        else:
            return float('inf')

    # Initial evaluation
    try:
        fa = _f(float(x0))
    except RuntimeError:
        return best_x

    if abs(fa) <= 1e-8:
        return float(x0)

    # Phase 1: Global Adaptive Bracketing
    # We look for a point b such that f(a)*f(b) < 0.
    # We expand the search range geometrically.
    a = float(x0)
    fa_val = fa
    
    # Initial step size
    h = 1e-4 * max(1.0, abs(a))
    
    bracket_found = False
    b = None
    fb = None
    
    # Try a series of offsets in both directions
    # The offsets are: +/- h, +/- 10h, +/- 100h, ..., +/- 10^8 h
    # This covers a wide range without wasting too many calls if the root is close.
    
    # To be efficient, we alternate directions or just try positive then negative.
    # Polynomials often have roots not too far from 0 or x0.
    
    search_steps = [
        h, -h,
        10*h, -10*h,
        100*h, -100*h,
        1000*h, -1000*h,
        10000*h, -10000*h,
        100000*h, -100000*h,
        1000000*h, -1000000*h,
        10000000*h, -10000000*h,
        100000000*h, -100000000*h
    ]
    
    # We also want to check if we've already passed a root in the expansion.
    # So we keep track of the previous point.
    prev_x = a
    prev_f = fa_val
    
    for offset in search_steps:
        if call_count >= max_calls - 10 or time.time() > deadline - 5.0:
            break
            
        x_try = a + offset
        try:
            f_try = _f(x_try)
        except RuntimeError:
            break
            
        if not math.isfinite(f_try):
            prev_x = x_try
            prev_f = f_try
            continue

        if math.isfinite(prev_f):
            if prev_f * f_try < 0:
                bracket_found = True
                a = prev_x
                b = x_try
                fa = prev_f
                fb = f_try
                break
            elif abs(f_try) <= 1e-8:
                return x_try
        
        # Update prev for next iteration to catch sign changes between non-adjacent test points
        # Actually, if we jump from x to x+100h, we might miss a root in between.
        # But for smooth polynomials, if f(x) and f(x+100h) have same sign, it's likely no root in between 
        # unless there are 2 roots. Given "3 fixed smooth polynomial equations", 
        # and typical task constraints, roots are usually simple or well-separated.
        # A more robust bracketing would use bisection-like expansion, but that costs calls.
        # Let's stick to the geometric expansion but update prev_x to the current test point 
        # only if we don't find a bracket, so we check the segment [prev_x, x_try] in the next step?
        # No, the list is fixed. 
        # Better strategy: 
        # 1. Check small offsets.
        # 2. If not found, expand exponentially.
        
        prev_x = x_try
        prev_f = f_try

    # If we didn't find a bracket with the geometric expansion, 
    # it's possible the root has even multiplicity (f touches 0 but doesn't cross) 
    # or x0 is very far.
    
    if not bracket_found:
        # Fallback: Local Minimization of |f(x)| using a simple pattern search / Nelder-Mead lite
        # This handles even multiplicity roots or cases where sign change is missed.
        
        # We use the best_x found so far as the starting point for local refinement.
        # We will try to minimize |f(x)| in the neighborhood of best_x.
        
        x_curr = best_x
        f_curr = best_f if best_f is not None else float('inf')
        
        # If we haven't found a good local min, try a few random-ish perturbations
        # But let's stick to a deterministic local search first.
        
        # Simple 1D pattern search
        step = 1e-3 * max(1.0, abs(x_curr))
        
        for _ in range(30):
            if call_count >= max_calls - 5 or time.time() > deadline - 5.0:
                break
                
            improved = False
            
            # Try expanding step if we are stuck
            # Try moving left and right
            
            x_left = x_curr - step
            x_right = x_curr + step
            
            f_left = _f(x_left)
            f_right = _f(x_right)
            
            if abs(f_left) < abs(f_curr):
                x_curr = x_left
                f_curr = f_left
                improved = True
                step *= 1.5 # Expand step if we made progress
            elif abs(f_right) < abs(f_curr):
                x_curr = x_right
                f_curr = f_right
                improved = True
                step *= 1.5
            else:
                # No improvement, shrink step
                step *= 0.5
                if step < 1e-15:
                    break
            
            if abs(f_curr) <= 1e-8:
                return x_curr

        # If local minimization didn't work well, try Secant method from best_x
        # Secant is often robust for polynomials if started close enough.
        
        x1 = x_curr
        f1 = f_curr
        h2 = 1e-5 * max(1.0, abs(x1))
        x2 = x1 + h2
        try:
            f2 = _f(x2)
        except:
            f2 = f1 + 1.0
            
        for _ in range(40):
            if call_count >= max_calls - 2 or time.time() > deadline - 2.0:
                break
            if abs(f1) <= 1e-8:
                return x1
            if abs(f2) <= 1e-8:
                return x2
            if f1 == f2:
                break
            x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
            if not math.isfinite(x_new):
                break
            try:
                f_new = _f(x_new)
            except:
                break
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new

        return best_x

    else:
        # Phase 2: Illinois/Regula Falsi Hybrid within [a, b]
        # Ensure fa and fb have opposite signs
        if fa * fb > 0:
            # Should not happen if bracket_found is True, but safety check
            # Try swapping or re-bracketing
            pass
            
        # Illinois method:
        # It's like Regula Falsi, but when the bracket is updated, 
        # the function value of the side that is retained is divided by 2 
        # to force convergence to bisection speed if one side is "stuck".
        
        x_lo = a
        x_hi = b
        f_lo = fa
        f_hi = fb
        
        for _ in range(60):
            if call_count >= max_calls - 2 or time.time() > deadline - 2.0:
                break
                
            # Regula Falsi step
            if f_lo == f_hi:
                x_mid = 0.5 * (x_lo + x_hi)
            else:
                x_mid = (x_lo * f_hi - x_hi * f_lo) / (f_hi - f_lo)
                
            # Ensure x_mid is within [x_lo, x_hi]
            if x_mid <= x_lo or x_mid >= x_hi:
                x_mid = 0.5 * (x_lo + x_hi)
                
            try:
                f_mid = _f(x_mid)
            except RuntimeError:
                break
                
            if abs(f_mid) <= 1e-8:
                return x_mid
                
            if f_lo * f_mid < 0:
                # Root is in [x_lo, x_mid]
                # Illinois: damp f_lo if x_hi was the one that got replaced? 
                # Standard Illinois: 
                # If we replace x_hi with x_mid, we set f_hi = f_hi / 2.
                # If we replace x_lo with x_mid, we set f_lo = f_lo / 2.
                
                # Here, x_hi is replaced by x_mid.
                x_hi = x_mid
                f_hi = f_mid
                # Damp the function value of the point that is being kept? 
                # No, Illinois method damps the value of the point that is *not* being replaced?
                # Let's check: 
                # "The Illinois method... divides the function value at the point that is *retained* by 2."
                # Wait, standard description:
                # If f(a)*f(c) < 0, then b is replaced by c. f(b) is replaced by f(b)/2.
                # So the point that is *removed* (b) has its value halved? No, b is removed.
                # The point that is *retained* (a) keeps its value?
                # Actually, the standard Illinois algorithm:
                # 1. Compute x_mid.
                # 2. If f(x_lo)*f(x_mid) < 0:
                #      x_hi = x_mid
                #      f_hi = f_hi / 2.0  <-- Damp the value of the point that is being REPLACED? 
                #      No, x_hi is replaced. So f_hi (old) is discarded. 
                #      The new f_hi is f_mid. 
                #      The damping is applied to the *other* side to force the next midpoint closer to the "stuck" side.
                #      
                # Let's use a more robust variant: 
                # If we update x_hi, we damp f_lo? Or f_hi?
                # Wikipedia: "The Illinois method... f(b) is replaced by f(b)/2".
                # If b is the upper bound and we replace it, we damp the *new* upper bound? No.
                # Let's look at the logic: We want to force the next step to be closer to the side that is "stuck" (the side where the function value hasn't changed sign relative to the other end in a useful way, i.e., the "flat" side).
                # If f(a) and f(c) have the same sign, the root is between c and b.
                # We replace a with c.
                # To prevent b from staying far away, we damp f(b).
                # So: If we replace x_lo, we damp f_hi. If we replace x_hi, we damp f_lo.
                
                # In this branch: f_lo * f_mid < 0. Root in [x_lo, x_mid].
                # We replace x_hi with x_mid.
                # We should damp f_lo.
                f_lo = f_lo / 2.0
                
            else:
                # Root is in [x_mid, x_hi]
                # We replace x_lo with x_mid.
                # We should damp f_hi.
                x_lo = x_mid
                f_lo = f_mid
                f_hi = f_hi / 2.0
            
            # Check convergence based on interval width
            if abs(x_hi - x_lo) < 1e-12 * max(1.0, abs(x_mid)):
                break

        # Return the point with the smallest |f| in the final bracket or best_x
        return best_x
# EVOLVE-BLOCK-END
