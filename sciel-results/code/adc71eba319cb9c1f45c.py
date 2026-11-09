"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a multi-start strategy with Damped Newton's Method.
    
    1. Generates multiple candidate starting points around x0.
    2. For each candidate, applies Damped Newton's Method with backtracking.
    3. Returns the first solution found.
    """
    x0 = float(x0)
    tol = 1e-8
    max_total_evals = 500
    eval_count = 0

    def f_count(x):
        nonlocal eval_count
        if eval_count >= max_total_evals:
            return 0.0 # Signal failure/stop
        eval_count += 1
        return f(x)

    # First check x0
    f0 = f_count(x0)
    if abs(f0) <= tol:
        return x0

    # Helper: Damped Newton's Method
    def newton_damped(x_start, max_iter=50):
        nonlocal eval_count
        x = x_start
        fx = f_count(x)
        
        if abs(fx) <= tol:
            return x, True

        # Estimate derivative using central difference
        # Use a small h relative to x
        h = 1e-5 * max(1.0, abs(x))
        
        for i in range(max_iter):
            if eval_count >= max_total_evals:
                return x, False
            
            # Calculate derivative
            # We need to be careful with evals here. 
            # Central diff costs 2 evals. 
            # We can reuse fx if we haven't moved, but here we move.
            # Optimization: If we just evaluated fx, we can use forward diff (1 eval) or central (2 evals).
            # Let's use central diff for stability, but it's expensive.
            # Alternative: Use secant-like update for derivative? 
            # Actually, for ill-conditioned, Newton is risky if derivative is small.
            # Let's stick to a robust derivative estimate.
            
            if h == 0:
                h = 1e-8
            fx_plus = f_count(x + h)
            if eval_count >= max_total_evals:
                return x, False
            fx_minus = f_count(x - h)
            if eval_count >= max_total_evals:
                return x, False
                
            dfx = (fx_plus - fx_minus) / (2 * h)
            
            if abs(dfx) < 1e-12:
                # Derivative is too small, Newton step is huge/unstable.
                # Try a small random kick or break
                break
                
            # Newton step
            dx = fx / dfx
            
            # Damping: Start with full step, backtrack if |f| increases
            step_scale = 1.0
            x_new = x - step_scale * dx
            fx_new = f_count(x_new)
            
            # Backtracking line search
            for _ in range(10):
                if eval_count >= max_total_evals:
                    return x, False
                if abs(fx_new) <= abs(fx):
                    break
                step_scale *= 0.5
                x_new = x - step_scale * dx
                fx_new = f_count(x_new)
            
            if abs(fx_new) <= tol:
                return x_new, True
            
            # Update
            if abs(fx_new) >= abs(fx):
                # If no improvement even with damping, this direction is bad.
                # Break to try another start point.
                break
                
            x = x_new
            fx = fx_new
            
            # Convergence check on step size
            if abs(dx) < 1e-10 * max(1.0, abs(x)):
                if abs(fx) <= tol:
                    return x, True
                # If step is tiny but f is not small enough, we are stuck.
                break
                
        return x, False

    # Generate candidate starting points
    # 1. x0 itself (already checked f0, but Newton might move it)
    # 2. Points at various scales around x0
    # We want to cover local neighborhood and slightly wider area.
    
    candidates = [x0]
    base_scale = max(1.0, abs(x0))
    
    # Logarithmic spacing of perturbations
    for exp in range(-8, 3):
        h = base_scale * (10 ** exp)
        candidates.append(x0 + h)
        candidates.append(x0 - h)
        
    # Remove duplicates and x0 (already handled, but Newton from x0 is valid)
    # Actually, let's just iterate.
    
    best_x = x0
    best_f = abs(f0)
    
    # Try Newton from x0 first
    x_sol, success = newton_damped(x0)
    if success:
        return x_sol
    if abs(f(x_sol)) < best_f: # Note: this uses an extra eval, need to be careful
        # We don't want to waste evals checking if we already have a good one.
        # Let's just keep the x from the failed attempt if it's closer?
        # The newton_damped returns the last x.
        pass

    # Try other candidates
    for x_cand in candidates:
        if x_cand == x0:
            continue
        if eval_count >= max_total_evals - 10: # Leave some buffer
            break
            
        # Quick check if we are already close
        # We don't want to eval f(x_cand) just to check, Newton will do it.
        # But if x_cand is far, Newton might waste evals.
        
        x_sol, success = newton_damped(x_cand)
        if success:
            return x_sol
            
        # Track best so far?
        # To track best, we need f(x_sol). 
        # The newton_damped function consumed evals.
        # We can check the last fx? No, it's local.
        # Let's just return the first success. If none, return x0.
        # To be safer, we can store the x with the smallest |f| seen in the last step of newton?
        # Modifying newton_damped to return best_x is complex.
        
    return best_x

# EVOLVE-BLOCK-END
