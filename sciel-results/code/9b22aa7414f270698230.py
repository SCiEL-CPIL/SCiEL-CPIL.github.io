"""Root finding: Ill-conditioned - Deterministic Hybrid Secant/Illinois Solver."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    
    Strategy:
    1. Robust bracketing search to establish a safe interval [a, b] with sign change.
    2. If bracket found, run Illinois algorithm (fast, guaranteed convergence within bracket).
    3. If no bracket found (or bracketing fails), run Secant method with aggressive damping.
    4. If Secant stalls or diverges, fallback to local minimization of |f(x)| via golden section
       or simple local search if necessary, but primarily rely on Secant/Illinois.
    
    This avoids the overhead of maintaining multiple candidate trajectories and complex scoring,
    focusing instead on efficient use of evaluations via Secant (reusing f values) and 
    guaranteed convergence via Illinois when possible.
    """
    start_time = time.time()
    MAX_WALL = 118.0  # Safety margin before 120s limit (evaluator kills at 150s, but task says 120s per problem? 
                      # "Each evaluation must finish within 120 s" likely means total time per problem. 
                      # "Kills program after 150s wall-clock" is the hard stop. 
                      # Let's use 119s to be safe against any overhead.)
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    eval_count = 0
    
    def eval_f(x):
        nonlocal eval_count
        eval_count += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('inf')
            return val
        except:
            return float('inf')

    def time_ok():
        return time.time() - start_time < MAX_WALL

    # Initial evaluation
    x0 = float(x0)
    fx0 = eval_f(x0)
    
    if abs(fx0) <= TARGET_TOL:
        return x0
    
    # --- Phase 1: Try to find a bracket ---
    # Expand from x0 to find a sign change.
    # Use geometric expansion to cover large ranges quickly.
    a, b, fa, fb = None, None, None, None
    h = max(1e-6, abs(x0) * 1e-6)
    found_bracket = False
    
    # We have limited evaluations. Let's spend at most ~20 on bracketing.
    # If we find it, Illinois is very fast. If not, we switch to Secant.
    bracket_budget = 20
    used_for_bracket = 0
    
    # Check immediate neighbors first with small steps
    for scale in [1, 10, 100, 1000, 1e4, 1e5, 1e6, 1e8, 1e10, 1e12]:
        if used_for_bracket >= bracket_budget or not time_ok():
            break
        h = max(1e-12, abs(x0) * 1e-6 * scale)
        if h < 1e-12:
            h = 1e-12
        if h > 1e12:
            h = 1e12
            
        for sign in [1, -1]:
            if used_for_bracket >= bracket_budget or not time_ok():
                break
            x_c = x0 + sign * h
            if x_c == x0:
                continue
            fx_c = eval_f(x_c)
            used_for_bracket += 1
            
            if abs(fx_c) <= TARGET_TOL:
                return x_c
            
            if fx0 * fx_c < 0:
                # Found bracket
                if x0 < x_c:
                    a, b, fa, fb = x0, x_c, fx0, fx_c
                else:
                    a, b, fa, fb = x_c, x0, fx_c, fx0
                found_bracket = True
                break
        if found_bracket:
            break

    # --- Phase 2: Solve ---
    best_x = x0
    best_fx = fx0
    
    if found_bracket:
        # Use Illinois Algorithm for guaranteed convergence within bracket
        # Illinois is faster than Bisection and more robust than Secant in a bracket.
        # It maintains a bracket [a, b] and picks the next point based on a weighted average.
        
        # Ensure |fa| < |fb| initially for standard Illinois logic, or just track them.
        # Standard Illinois:
        # 1. Start with [a, b] such that f(a)f(b) < 0.
        # 2. Pick x1 = (a + b) / 2.
        # 3. If f(x1) is closer to 0 than f(a), set b = x1. Else set a = x1.
        # 4. If we moved a, halve f(b) (or the value at the endpoint we didn't move) to accelerate convergence away from the slower side.
        
        # Let's implement a robust Illinois.
        # State: a, b, fa, fb
        # We want to minimize |f(x)|.
        
        # Initial step: midpoint
        x_mid = (a + b) / 2
        fx_mid = eval_f(x_mid)
        
        if abs(fx_mid) <= TARGET_TOL:
            return x_mid
        
        # Update bracket
        if fa * fx_mid < 0:
            b, fb = x_mid, fx_mid
        else:
            a, fa = x_mid, fx_mid
        
        # Now iterate
        while time_ok() and eval_count < MAX_EVALS:
            # Check convergence
            if abs(fa) <= TARGET_TOL:
                return a
            if abs(fb) <= TARGET_TOL:
                return b
            
            # Illinois step
            # If |fa| < |fb|, the root is likely closer to a.
            # We want to move b closer to a.
            # The standard Illinois formula for the new point x is:
            # If we keep a and move b: x = a - fa * (b - a) / (fb - fa)
            # But we also halve the function value of the endpoint we keep to force the next step to be closer.
            
            # Let's follow the standard algorithm:
            # 1. If |fa| < |fb|:
            #    x = a - fa * (b - a) / (fb - fa)
            #    Evaluate f(x)
            #    If f(a) and f(x) have same sign:
            #       a = x, fa = f(x)
            #       fb = 0.5 * fb  (Halve the value at b)
            #    Else:
            #       b = x, fb = f(x)
            # 2. Else (|fb| <= |fa|):
            #    x = b - fb * (a - b) / (fa - fb)
            #    Evaluate f(x)
            #    If f(b) and f(x) have same sign:
            #       b = x, fb = f(x)
            #       fa = 0.5 * fa
            #    Else:
            #       a = x, fa = f(x)
            
            if abs(fa) < abs(fb):
                denom = fb - fa
                if denom == 0:
                    x = (a + b) / 2
                else:
                    x = a - fa * (b - a) / denom
                
                # Ensure x is strictly inside (a, b)
                if x <= a or x >= b:
                    x = (a + b) / 2
                
                fx = eval_f(x)
                
                if abs(fx) <= TARGET_TOL:
                    return x
                
                if fa * fx < 0:
                    # Root is between a and x
                    b, fb = x, fx
                else:
                    # Root is between x and b
                    a, fa = x, fx
                    fb *= 0.5  # Halve the function value at the other endpoint
            else:
                denom = fa - fb
                if denom == 0:
                    x = (a + b) / 2
                else:
                    x = b - fb * (a - b) / denom
                
                if x <= a or x >= b:
                    x = (a + b) / 2
                
                fx = eval_f(x)
                
                if abs(fx) <= TARGET_TOL:
                    return x
                
                if fb * fx < 0:
                    # Root is between b and x
                    a, fa = x, fx
                else:
                    # Root is between a and x
                    b, fb = x, fx
                    fa *= 0.5
            
            # Update best
            if abs(fa) < abs(best_fx):
                best_x, best_fx = a, fa
            if abs(fb) < abs(best_fx):
                best_x, best_fx = b, fb

    else:
        # No bracket found. Use Secant Method.
        # Secant requires two points. We have x0, fx0.
        # We need x1, fx1.
        # Let's pick x1 = x0 + h where h is small.
        h = max(1e-5, abs(x0) * 1e-5)
        x1 = x0 + h
        if x1 == x0:
            x1 = x0 + 1e-5
        fx1 = eval_f(x1)
        
        if abs(fx1) <= TARGET_TOL:
            return x1
        
        # Secant iteration
        x_a, fx_a = x0, fx0
        x_b, fx_b = x1, fx1
        
        # Track progress to detect stagnation
        prev_best = min(abs(fx_a), abs(fx_b))
        stall_count = 0
        
        while time_ok() and eval_count < MAX_EVALS:
            if abs(fx_a) <= TARGET_TOL:
                return x_a
            if abs(fx_b) <= TARGET_TOL:
                return x_b
            
            # Secant step
            denom = fx_b - fx_a
            if denom == 0 or not math.isfinite(denom):
                # Break tie, try a different step
                x_b = x_a + 1e-4
                fx_b = eval_f(x_b)
                continue
            
            x_new = x_a - fx_a * (x_b - x_a) / denom
            
            # Damping: if step is too large, reduce it
            max_step = 10 * max(1.0, abs(x_a))
            if abs(x_new - x_a) > max_step:
                x_new = x_a + math.copysign(max_step, x_new - x_a)
            
            if not math.isfinite(x_new):
                break
                
            fx_new = eval_f(x_new)
            
            if abs(fx_new) <= TARGET_TOL:
                return x_new
            
            # Update points
            x_a, fx_a = x_b, fx_b
            x_b, fx_b = x_new, fx_new
            
            # Check progress
            current_best = min(abs(fx_a), abs(fx_b))
            if current_best < prev_best * 0.9:
                prev_best = current_best
                stall_count = 0
            else:
                stall_count += 1
            
            # If stalled, try a local perturbation or switch to a different strategy?
            # For simplicity, if stalled for a while, we might just return the best found.
            # But let's try to "kick" it by taking a smaller step if it's oscillating.
            if stall_count > 5:
                # Try a bisection-like step between the two worst points?
                # Or just perturb x_b slightly
                x_b = x_a + 0.5 * (x_b - x_a)
                fx_b = eval_f(x_b)
                stall_count = 0
                
            # Update best
            if abs(fx_a) < abs(best_fx):
                best_x, best_fx = x_a, fx_a
            if abs(fx_b) < abs(best_fx):
                best_x, best_fx = x_b, fx_b

    return best_x

# EVOLVE-BLOCK-END
