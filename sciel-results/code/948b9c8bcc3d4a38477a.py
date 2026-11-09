"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Strategy: 
    1. Global grid scan to find a sign change (bracket) using geometric spacing.
    2. Apply Brent's method on the first valid bracket found.
    3. Fallback to best point found during scan if no bracket or Brent fails.
    
    This approach is structurally different from local secant methods because
    it prioritizes finding a guaranteed bracket via global sampling, which is
    robust for oscillatory functions where local derivatives might be misleading.
    """
    deadline = time.time() + 145.0
    tol = 1e-12  # Internal tolerance slightly tighter than 1e-8 to be safe
    x0 = float(x0)
    
    best_x = x0
    best_f = f(x0)
    best_abs = abs(best_f) if math.isfinite(best_f) else float('inf')
    
    if best_abs <= 1e-8:
        return best_x

    cache = {x0: best_f}
    
    def eval_f(x_val):
        nonlocal best_x, best_f, best_abs
        if x_val in cache:
            f_val = cache[x_val]
        else:
            try:
                f_val = f(x_val)
            except Exception:
                f_val = float('nan')
            cache[x_val] = f_val
        
        if f_val is None or not math.isfinite(f_val) or not math.isfinite(x_val):
            return f_val
        
        abs_f = abs(f_val)
        if abs_f < best_abs:
            best_x = x_val
            best_f = f_val
            best_abs = abs_f
            
        return f_val

    def brent_method(a, b, fa, fb, max_iter=60):
        """
        Implement Brent's method (or a robust mix of Bisection/Secant/Inverse Quadratic)
        to find a root in [a, b] where f(a)*f(b) < 0.
        Returns x such that |f(x)| < tol, or None if it fails.
        """
        # Variables for Brent's method
        x0 = a
        x1 = b
        x2 = b
        f0 = fa
        f1 = fb
        f2 = fb
        a, b = a, b
        
        for i in range(max_iter):
            if time.time() > deadline:
                return None
                
            # Check for convergence
            if abs(f1) < tol:
                return x1
            if abs(f0) < tol:
                return x0
            
            # Inverse quadratic interpolation
            # We need 3 points. We have x0, x1, x2.
            # Note: Standard Brent uses x0 (previous), x1 (current best), x2 (other end)
            # Let's use a simplified robust version: Secant + Bisection fallback
            
            # Try Secant
            if f0 != f1:
                s = x1 - f1 * (x1 - x0) / (f1 - f0)
            else:
                s = None
            
            # Conditions for using Secant (s)
            # 1. s must be within [min(a,b), max(a,b)] or close
            # 2. s must not be too close to x1
            # 3. Bisection condition: if secant is bad, use bisection
            
            # Standard Brent's conditions:
            # Let's stick to a simpler robust "Illinois" or "Regula Falsi" variant 
            # or just pure Bisection if we want to be safe, but Secant is faster.
            
            # Let's use a modified Illinois method which is very robust for oscillatory functions
            # and converges linearly but reliably, using fewer calls than pure bisection.
            
            # Illinois Algorithm:
            # 1. Calculate s (secant)
            # 2. If s is outside [a,b] or too close to x1, use bisection
            # 3. If s is accepted, and f(s) has same sign as f(x1), 
            #    then replace the OTHER point with a weighted average (Illinois damping)
            
            # Determine which end to keep
            # We want to keep the point with smaller |f| as x1 (current best)
            if abs(f0) < abs(f1):
                # Swap so x1 is the best
                x0, x1 = x1, x0
                f0, f1 = f1, f0
                # Also swap a, b conceptually? No, a,b are the bracket bounds.
                # x0, x1 are the two most recent points.
                # The bracket is still [a, b].
                # We need to know which of a,b corresponds to x0, x1.
                # This gets complicated. Let's just implement a standard Bisection+Secant hybrid.
                
            # Simpler approach: Standard Bisection with Secant acceleration
            # If secant step is within bracket, take it. Else bisect.
            
            if s is not None and math.isfinite(s):
                # Check if s is in bracket [a, b]
                if (a < s < b) or (b < s < a):
                    # Also ensure it's not too close to x1
                    if abs(s - x1) > 1e-15 * max(1.0, abs(x1)):
                        # Accept secant step
                        x_new = s
                        f_new = eval_f(x_new)
                        
                        if not math.isfinite(f_new):
                            # Fallback to bisection
                            x_new = (a + b) / 2.0
                            f_new = eval_f(x_new)
                    else:
                        x_new = (a + b) / 2.0
                        f_new = eval_f(x_new)
                else:
                    # Secant step outside, use bisection
                    x_new = (a + b) / 2.0
                    f_new = eval_f(x_new)
            else:
                x_new = (a + b) / 2.0
                f_new = eval_f(x_new)

            if not math.isfinite(f_new):
                return None

            # Update bracket
            if f1 * f_new < 0:
                # Root is between x1 and x_new
                a = min(x1, x_new)
                b = max(x1, x_new)
                # x0 becomes x1? No, in this simple hybrid, we just keep the bracket.
                # For the next secant step, we need two points.
                # Let's keep x0 = x1, x1 = x_new?
                # Actually, standard secant needs the two most recent.
                # Let's update x0, x1 to be the endpoints of the new bracket?
                # No, that's not secant.
                
                # Let's refine the state:
                # We maintain x0, x1 as the two most recent points used for secant.
                # But we also maintain a, b as the bracket.
                
                # If f1 and f_new have opposite signs, the root is in [x1, x_new].
                # So a, b = x1, x_new (ordered).
                # The two points for the next secant step should be the ones bounding the root?
                # Or the two most recent evaluations?
                # Usually, we use the two points that define the current bracket for the next secant.
                # So x0 = a, x1 = b? No, f(a) and f(b) are known.
                
                # Let's set x0, x1 to the new bracket endpoints for the next iteration's secant calculation.
                # This effectively makes it a "Regula Falsi" variant if we always use the bracket endpoints.
                # But pure Regula Falsi can be slow. 
                # To accelerate, we could keep the other point.
                # Given the low call budget, let's stick to a robust Bisection/Secant mix
                # where we use the current bracket endpoints for the secant estimate.
                
                # Update a, b
                if x1 < x_new:
                    a, b = x1, x_new
                else:
                    a, b = x_new, x1
                    
                # Update x0, x1 for next secant step to be the new bracket endpoints
                # This is safe and robust.
                x0, x1 = a, b
                f0, f1 = f(a) if a in cache else eval_f(a), f(b) if b in cache else eval_f(b)
                
                # Wait, eval_f updates best_x. We shouldn't re-eval if in cache.
                # But we need f0, f1 values.
                # If a == old x1, f0 = old f1. If a == x_new, f0 = f_new.
                # Let's just track values directly to avoid confusion.
                
                if abs(f_new) < tol:
                    return x_new
                    
            else:
                # f1 and f_new have same sign. Root is between x0 (other end of previous bracket?) 
                # This logic is getting messy with the simple a,b,x0,x1 state.
                
                # Let's restart the root finding loop with a cleaner state machine.
                # Actually, let's just use a dedicated function for the bracket refinement
                # that maintains clear state.
                
                pass # This block is replaced by the cleaner implementation below

        return None

    # --- Cleaner Implementation of Bracket Refinement ---
    
    def refine_bracket(a, b, fa, fb, max_iter=50):
        """
        Find root in [a,b] given f(a)*f(b)<0.
        Uses Bisection/Secant hybrid.
        """
        x0 = a
        x1 = b
        f0 = fa
        f1 = fb
        
        for _ in range(max_iter):
            if time.time() > deadline:
                return None
                
            if abs(f0) < tol:
                return x0
            if abs(f1) < tol:
                return x1
            
            # Secant step
            denom = f1 - f0
            if denom == 0 or not math.isfinite(denom):
                s = (x0 + x1) / 2.0
            else:
                s = x1 - f1 * (x1 - x0) / denom
                
            # Check if s is in bracket [x0, x1]
            in_bracket = False
            if x0 < x1:
                in_bracket = (x0 < s < x1)
            else:
                in_bracket = (x1 < s < x0)
                
            # If s is not in bracket, or too close to x1, use bisection
            if not in_bracket or abs(s - x1) < 1e-15 * max(1.0, abs(x1)):
                s = (x0 + x1) / 2.0
                
            fs = eval_f(s)
            if not math.isfinite(fs):
                return None
                
            if abs(fs) < tol:
                return s
                
            # Update bracket
            if f0 * fs < 0:
                # Root in [x0, s]
                x1 = s
                f1 = fs
            else:
                # Root in [s, x1]
                x0 = s
                f0 = fs
                
        return None

    # --- Global Grid Scan ---
    # We scan with geometrically increasing steps to find a sign change.
    # Start with a small step to catch local roots, then expand.
    
    # Initial step size
    # Heuristic: 1e-3 is a good start for general functions, but for oscillatory,
    # we might need smaller or larger.
    # Let's try a few base steps.
    
    # Strategy: 
    # 1. Try a local scan around x0 with small steps.
    # 2. If no root found, expand the range.
    
    # To minimize calls, we will do a "spiral" or "expanding" scan.
    # Points: x0, x0+d, x0+2d, x0+4d... and x0-d, x0-2d...
    # But we need to check sign changes between consecutive points.
    
    # Let's define a list of points to evaluate in a specific order.
    # Order: 
    # 1. x0 (already done)
    # 2. x0 + d1
    # 3. x0 - d1
    # 4. x0 + 2*d1
    # 5. x0 - 2*d1
    # ...
    # 6. x0 + 4*d1
    # 7. x0 - 4*d1
    # ...
    
    # d1 = 1e-3 initially? Or 0.1?
    # The cell finding said "Coarser initial step (0.1 vs 0.02) means fewer calls".
    # Let's use d1 = 0.1.
    
    d = 0.1
    # We will expand d by factor 2 each time we double the step size.
    # But we also need to check intermediate points?
    # If we jump from x0 to x0+0.1, we might miss a root if the period is < 0.1.
    # However, the task says "oscillatory". If the period is very small, we need small steps.
    # But if we use small steps, we use more calls.
    # The previous program used scales from 1e-5 to 1e4.
    
    # Let's try a hybrid:
    # First, check a few small steps (1e-3, 1e-2) to catch high-frequency oscillations.
    # Then expand geometrically.
    
    steps = [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    # We will iterate through these scales.
    # For each scale, we check points x0 + k*scale and x0 - k*scale for k=1, 2, 4, 8...
    # But we must ensure we check sign changes between the last known point and the new one.
    
    # Let's maintain a list of "checked" intervals.
    # Actually, simpler:
    # For each scale s:
    #   Check x0 + s, x0 - s
    #   If sign change with x0, refine.
    #   If no, check x0 + 2s, x0 - 2s
    #   If sign change with x0+s or x0-s, refine.
    #   And so on.
    
    # This might be inefficient if we re-check.
    # Let's just generate a sequence of points and check consecutive pairs.
    
    # Sequence:
    # x0
    # x0 + 0.001, x0 - 0.001
    # x0 + 0.002, x0 - 0.002
    # x0 + 0.004, x0 - 0.004
    # ...
    # x0 + 0.01, x0 - 0.01
    # ...
    # This is getting complicated to manage state.
    
    # Alternative: Just use a fixed set of "bracket candidates" generated by geometric expansion.
    # For each expansion level L (0 to 10):
    #   step = 0.1 * (2 ** L)
    #   Check points: x0 + step, x0 - step
    #   Check sign change between (x0, x0+step) and (x0, x0-step)
    #   Check sign change between (x0+step, x0+2*step) ? No, we didn't check 2*step yet.
    
    # Let's stick to the previous program's success: it used multiple scales.
    # The key improvement is using Brent/Refine on the FIRST bracket found.
    # The previous program tried Secant on MANY brackets.
    # We will try to find ONE good bracket and solve it efficiently.
    
    # To find a bracket, we scan.
    # Scan order:
    # 1. Local: x0 +/- 1e-3, +/- 1e-2
    # 2. Medium: x0 +/- 0.1, +/- 1.0
    # 3. Large: x0 +/- 10, +/- 100
    
    # We will check consecutive points in this ordered list.
    # List:
    # [x0]
    # Add x0+0.001, x0-0.001
    # Add x0+0.01, x0-0.01
    # Add x0+0.1, x0-0.1
    # Add x0+1, x0-1
    # Add x0+10, x0-10
    # Add x0+100, x0-100
    
    # But we need to check sign changes between ADJACENT points in the sorted order?
    # No, just between the new point and the closest previously checked point.
    
    # Let's implement a simple expanding scan.
    
    # Initialize with x0
    # prev_pos = x0, prev_neg = x0
    
    # We'll keep track of the last positive and last negative points.
    # If we find a new point with opposite sign to one of them, we have a bracket.
    
    last_pos_x = x0
    last_pos_f = best_f
    last_neg_x = x0
    last_neg_f = best_f
    
    # If best_f is 0, we already returned.
    # So best_f is either positive or negative.
    if best_f > 0:
        last_pos_x = x0
        last_pos_f = best_f
    else:
        last_neg_x = x0
        last_neg_f = best_f

    # Define scan points
    # We use a geometric progression of offsets.
    # Offsets: 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000
    # For each offset, we check +offset and -offset.
    
    offsets = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    
    found_bracket = False
    bracket_a, bracket_b = 0, 0
    bracket_fa, bracket_fb = 0, 0
    
    for off in offsets:
        if time.time() > deadline:
            break
            
        # Check x0 + off
        x_plus = x0 + off
        f_plus = eval_f(x_plus)
        
        if math.isfinite(f_plus):
            if f_plus > 0:
                if last_neg_f != 0 and (last_neg_x < x_plus or last_neg_x > x_plus): # Always true if distinct
                    # Check if we have a bracket with last_neg
                    # The bracket is between last_neg_x and x_plus
                    # But we must ensure there isn't a closer point?
                    # For oscillatory functions, a bracket between non-adjacent points might contain multiple roots.
                    # Brent's method will find ONE root. That's fine.
                    bracket_a = last_neg_x
                    bracket_b = x_plus
                    bracket_fa = last_neg_f
                    bracket_fb = f_plus
                    found_bracket = True
                    break
                last_pos_x = x_plus
                last_pos_f = f_plus
            elif f_plus < 0:
                if last_pos_f != 0:
                    bracket_a = last_pos_x
                    bracket_b = x_plus
                    bracket_fa = last_pos_f
                    bracket_fb = f_plus
                    found_bracket = True
                    break
                last_neg_x = x_plus
                last_neg_f = f_plus
            else:
                return x_plus # Found exact root

        # Check x0 - off
        x_minus = x0 - off
        f_minus = eval_f(x_minus)
        
        if math.isfinite(f_minus):
            if f_minus > 0:
                if last_neg_f != 0:
                    bracket_a = last_neg_x
                    bracket_b = x_minus
                    bracket_fa = last_neg_f
                    bracket_fb = f_minus
                    found_bracket = True
                    break
                last_pos_x = x_minus
                last_pos_f = f_minus
            elif f_minus < 0:
                if last_pos_f != 0:
                    bracket_a = last_pos_x
                    bracket_b = x_minus
                    bracket_fa = last_pos_f
                    bracket_fb = f_minus
                    found_bracket = True
                    break
                last_neg_x = x_minus
                last_neg_f = f_minus
            else:
                return x_minus

    if found_bracket:
        # Refine the bracket
        # Ensure a < b
        if bracket_a > bracket_b:
            bracket_a, bracket_b = bracket_b, bracket_a
            bracket_fa, bracket_fb = bracket_fb, bracket_fa
            
        root = refine_bracket(bracket_a, bracket_b, bracket_fa, bracket_fb, max_iter=60)
        if root is not None and math.isfinite(root):
            # Verify
            f_root = eval_f(root)
            if math.isfinite(f_root) and abs(f_root) <= 1e-8:
                return root
            # If refinement didn't get it under 1e-8, return root anyway if it's better?
            # The contract says |f(x)| <= 1e-8.
            # If refine_bracket returned, it likely converged.
            # But let's check.
            if abs(f_root) < best_abs:
                return root
            # If it's not good enough, fall back to best_x

    # If no bracket found or refinement failed, return best_x
    return best_x

# EVOLVE-BLOCK-END
