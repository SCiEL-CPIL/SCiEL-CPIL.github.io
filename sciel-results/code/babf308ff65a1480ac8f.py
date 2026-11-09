"""Root finding for ill-conditioned equations: MODEL-AND-OPTIMIZE with adaptive refinement."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin below 120s limit
    MAX_EVALS = 480   # Safety margin below 500 limit

    # State tracking
    eval_count = 0
    best_x = float(x0)
    best_f = None
    best_abs = float('inf')
    
    # Store all evaluated points for potential re-use or as candidates
    eval_history = []  # list of (x, fval)

    try:
        best_f = f(best_x)
        eval_count += 1
        if math.isfinite(best_f):
            best_abs = abs(best_f)
            eval_history.append((best_x, best_f))
    except Exception:
        best_f = float('inf')
        best_abs = float('inf')

    def check_budget():
        nonlocal eval_count
        if eval_count > MAX_EVALS or (time.time() - start_time) > DEADLINE:
            return True
        return False

    def safe_eval(x):
        nonlocal eval_count, best_x, best_f, best_abs
        if check_budget():
            return None
        # Check if we already evaluated this point (exact match)
        # This is a simple O(N) check, but N is small enough (<500)
        # Optimizing this would require a dict, but floating point equality is tricky.
        # Let's just rely on the solver logic not re-evaluating too much.
        try:
            val = f(x)
            if not math.isfinite(val):
                return None
            eval_count += 1
            abs_val = abs(val)
            if abs_val < best_abs:
                best_f = val
                best_x = x
                best_abs = abs_val
            eval_history.append((x, val))
            return val
        except Exception:
            return None

    # 1. Initial Check
    if best_f is not None and best_abs <= 1e-8:
        return best_x

    # Strategy: "Model-and-Optimize"
    # The idea is to fit a local model (polynomial) to recent evaluations and optimize
    # that model to find a candidate for the next evaluation.
    # For root finding, we can use:
    # 1. Newton (linear model)
    # 2. Secant (linear model through two points)
    # 3. Steffensen (quadratic model via fixed point iteration x - f(x)^2/f(x+df))
    # 4. Inverse Interpolation (fit y=f(x), find x where y=0)

    # We will implement a robust solver that combines these.
    
    # Helper: Find a root starting from x_start using a hybrid method
    # Returns the best x found in this local search
    def solve_local(x_start, max_steps=20):
        nonlocal eval_count, best_x, best_f, best_abs
        
        x = float(x_start)
        fx = safe_eval(x)
        if fx is None:
            return None
            
        if abs(fx) <= 1e-8:
            return x

        # Maintain a small history of points for inverse interpolation
        # (x, f(x))
        # We'll use a list of the last few points
        
        # Initial step: try to establish a derivative or a second point
        # Use a small perturbation
        h = 1e-5 * max(1.0, abs(x))
        x2 = x + h
        fx2 = safe_eval(x2)
        
        if fx2 is None:
            return x
            
        if abs(fx2) < abs(fx):
            x, fx = x2, fx2
            # Need another point
            x2 = x - h
            fx2 = safe_eval(x2)
            if fx2 is None:
                return x
            if abs(fx2) < abs(fx):
                 x, fx = x2, fx2
        
        # Now we have two points: (x, fx) and (x2, fx2)
        # If they are the same, perturb x
        if x == x2:
             x2 = x + 1e-4
             fx2 = safe_eval(x2)
             if fx2 is None:
                 return x

        # Store the two points
        x1, f1 = x, fx
        x0_, f0_ = x2, fx2
        
        # Iterate
        for _ in range(max_steps):
            if check_budget():
                break
                
            # Check success
            if abs(f1) <= 1e-8:
                return x1
            if abs(f0_) <= 1e-8:
                return x0_

            # Method selection:
            # If we have a bracket, use Bisection or Regula Falsi.
            # If not, use Secant or Inverse Interpolation.
            
            if f1 * f0_ < 0:
                # Bracketed root. Use Illinois algorithm for faster convergence.
                bracket_width = abs(x1 - x0_)
                
                # Illinois: replace the endpoint with smaller |f|
                if abs(f0_) < abs(f1):
                    f0_ = f0_ / 2.0
                    x_new = (x0_ * f1 - x1 * f0_) / (f1 - f0_)
                    # Keep x_new within bracket
                    if x_new < min(x0_, x1) or x_new > max(x0_, x1):
                        x_new = (x0_ + x1) / 2.0
                else:
                    f1 = f1 / 2.0
                    x_new = (x0_ * f1 - x1 * f0_) / (f1 - f0_)
                    if x_new < min(x0_, x1) or x_new > max(x0_, x1):
                        x_new = (x0_ + x1) / 2.0

                f_new = safe_eval(x_new)
                if f_new is None:
                    break
                
                # Update bracket
                if f1 * f_new < 0:
                    x0_, f0_ = x1, f1
                    x1, f1 = x_new, f_new
                else:
                    x0_, f0_ = x_new, f_new
                    
                # Convergence check
                if abs(x1 - x0_) < 1e-12 * max(1.0, abs(x1)):
                    break

            else:
                # Unbracketed. Use Secant.
                if f1 == f0_:
                    # Degenerate, use small step
                    x1 = x1 + 1e-4
                    f1 = safe_eval(x1)
                    if f1 is None: break
                    continue
                
                # Secant step
                x_new = x1 - f1 * (x1 - x0_) / (f1 - f0_)
                
                # Safeguard: Don't jump too far (less restrictive)
                max_jump = max(1.0, 2.0 * abs(x1 - x0_))
                if abs(x_new - x1) > max_jump:
                    x_new = x1 + math.copysign(max_jump, x_new - x1)
                
                f_new = safe_eval(x_new)
                if f_new is None:
                    break
                
                # Keep the two points with smallest |f| for better convergence
                if abs(f_new) < abs(f0_):
                    x0_, f0_ = x_new, f_new
                    # Keep x1 as is
                else:
                    x0_, f0_ = x1, f1
                    x1, f1 = x_new, f_new
                
                # Stagnation
                if abs(x1 - x0_) < 1e-14:
                    break
        
        return x1

    # Main Loop
    # We will try a sequence of starting points.
    # 1. x0
    # 2. x0 +/- small perturbations
    # 3. x0 +/- larger perturbations
    # 4. Global sampling
    
    # Strategy 1: Direct local solve from x0
    result = solve_local(x0, max_steps=30)
    if result is not None and best_abs <= 1e-8:
        return best_x

    # Strategy 2: Perturbed starts
    # For ill-conditioned problems, the root might be very close but the function is steep.
    # Or the derivative is small, making Newton unstable.
    # Perturbations help find a direction.
    
    perturbations = [
        1e-2, -1e-2,
        1e-3, -1e-3,
        1e-1, -1e-1,
        0.5, -0.5,
        1.0, -1.0
    ]
    
    for p in perturbations:
        if check_budget():
            break
        x_pert = x0 + p
        result = solve_local(x_pert, max_steps=15)
        if result is not None and best_abs <= 1e-8:
            return best_x

    # Strategy 3: Broader sampling to find a sign change or better starting point
    # Sample points logarithmically spaced around x0
    # This helps if the root is far away.
    
    radii = [10.0, 100.0, 1000.0]
    for r in radii:
        if check_budget():
            break
        # Sample a few points
        # We want to cover [x0 - r, x0 + r]
        # Let's pick 4-6 points
        samples = [
            x0 - r,
            x0 - r/2,
            x0 + r/2,
            x0 + r,
            x0 - r*0.1,
            x0 + r*0.1
        ]
        
        for x_s in samples:
            if check_budget():
                break
            # Evaluate to update best
            fs = safe_eval(x_s)
            if fs is not None:
                if best_abs <= 1e-8:
                    return best_x
                # Try to solve from this point
                result = solve_local(x_s, max_steps=10)
                if result is not None and best_abs <= 1e-8:
                    return best_x

    # Strategy 4: Final polish pass with tighter tolerance
    # If we're close but not quite there, try a high-precision polish
    if best_abs < 1e-4:
        # Try secant polish from the best point with more iterations
        h = 1e-10 * max(1.0, abs(best_x))
        x1, f1 = best_x, best_f
        x2 = best_x + h
        f2 = safe_eval(x2)
        if f2 is not None:
            for _ in range(30):
                if check_budget():
                    break
                if abs(f1) <= 1e-8:
                    return x1
                if abs(f2) <= 1e-8:
                    return x2
                if abs(f1 - f2) < 1e-30:
                    break
                x_new = x1 - f1 * (x1 - x2) / (f1 - f2)
                f_new = safe_eval(x_new)
                if f_new is None:
                    break
                # Keep the two best points
                if abs(f_new) < abs(f2):
                    x2, f2 = x_new, f_new
                else:
                    x1, f1 = x2, f2
                    x2, f2 = x_new, f_new
                if abs(x1 - x2) < 1e-15 * max(1.0, abs(x1)):
                    break
        
        # Strategy 4b: If still not close enough, try a different perturbation
        if best_abs > 1e-8:
            h2 = 1e-8 * max(1.0, abs(best_x))
            x1, f1 = best_x, best_f
            x2 = best_x - h2
            f2 = safe_eval(x2)
            if f2 is not None and abs(f2) < best_abs:
                for _ in range(20):
                    if check_budget():
                        break
                    if abs(f1) <= 1e-8:
                        return x1
                    if abs(f2) <= 1e-8:
                        return x2
                    if abs(f1 - f2) < 1e-30:
                        break
                    x_new = x1 - f1 * (x1 - x2) / (f1 - f2)
                    f_new = safe_eval(x_new)
                    if f_new is None:
                        break
                    if abs(f_new) < abs(f2):
                        x2, f2 = x_new, f_new
                    else:
                        x1, f1 = x2, f2
                        x2, f2 = x_new, f_new
                    if abs(x1 - x2) < 1e-15 * max(1.0, abs(x1)):
                        break

    # Strategy 5: If all else fails, return the best x found so far
    # This maximizes the chance of being close even if not within 1e-8
    return best_x

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-END
