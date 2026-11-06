import time
import math
from scipy.optimize import brentq, minimize_scalar

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    max_evals = 480  # 20 evals safety margin
    eval_count = 0
    
    def eval_f(x):
        nonlocal eval_count, start_time
        if eval_count >= max_evals:
            return float('inf')
        if time.time() > deadline:
            return float('inf')
        eval_count += 1
        try:
            return f(x)
        except:
            return float('inf')

    x0 = float(x0)
    fx0 = eval_f(x0)
    if math.isnan(fx0) or math.isinf(fx0):
        # Try a small perturbation
        x0 = x0 + 1e-6
        fx0 = eval_f(x0)
    
    if abs(fx0) <= 1e-8:
        return x0

    # Strategy 1: Try to find a bracket and use Brent's method
    # Expand interval around x0
    h = 1e-3
    a, b = None, None
    fa, fb = None, None
    
    # We want to find a sign change
    # Start with a small step and expand
    for i in range(20):
        if time.time() > deadline or eval_count >= max_evals:
            break
        xa = x0 - h
        xb = x0 + h
        fa = eval_f(xa)
        fb = eval_f(xb)
        
        if math.isnan(fa) or math.isinf(fa) or math.isnan(fb) or math.isinf(fb):
            h *= 2
            continue
            
        if abs(fa) <= 1e-8:
            return xa
        if abs(fb) <= 1e-8:
            return xb
            
        if fa * fb < 0:
            a, b = xa, xb
            break
        h *= 2.0

    if a is not None:
        try:
            # Use brentq with tight tolerance
            # xtol=1e-12, rtol=1e-12
            root = brentq(eval_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if abs(eval_f(root)) <= 1e-8:
                return root
        except Exception:
            pass

    # Strategy 2: Minimize f(x)^2 using Brent's method for minimization
    # This works even without a sign change (e.g., tangent roots)
    def objective(x):
        fx = eval_f(x)
        if math.isnan(fx) or math.isinf(fx):
            return 1e100
        return fx * fx

    # Define a reasonable search interval
    # We already expanded h in the bracket search, use that range or a default
    search_a = x0 - max(1.0, abs(x0)) * 2.0
    search_b = x0 + max(1.0, abs(x0)) * 2.0
    
    # If we found a bracket in Strategy 1, we might still be in a local min of f^2
    # But if no bracket was found, we minimize f^2
    
    try:
        res = minimize_scalar(objective, bounds=(search_a, search_b), method='brent', options={'xatol': 1e-12, 'maxiter': 100})
        if res.success:
            x_opt = res.x
            fx_opt = eval_f(x_opt)
            if abs(fx_opt) <= 1e-8:
                return x_opt
    except Exception:
        pass

    # Strategy 3: Fallback - Secant/Newton hybrid with finite difference
    # Use the best point found so far
    x_best = x0
    fx_best = fx0
    if 'x_opt' in locals() and abs(eval_f(x_opt)) < abs(fx_best):
        x_best = x_opt
        fx_best = eval_f(x_best)

    # Perform a few secant steps from the best point
    h = 1e-6 * max(1.0, abs(x_best))
    x_prev = x_best - h
    fx_prev = eval_f(x_prev)
    
    for _ in range(50):
        if time.time() > deadline or eval_count >= max_evals:
            break
        if abs(fx_best) < abs(fx_prev):
            x_prev, x_best = x_best, x_prev
            fx_prev, fx_best = fx_best, fx_prev
            
        if abs(fx_best) <= 1e-8:
            return x_best
            
        if fx_best == fx_prev:
            break
            
        x_new = x_best - fx_best * (x_best - x_prev) / (fx_best - fx_prev)
        
        # Damping if step is too large
        if abs(x_new - x_best) > 10 * max(1.0, abs(x_best)):
            x_new = x_best + 0.5 * (x_new - x_best)
            
        fx_new = eval_f(x_new)
        
        if abs(fx_new) < abs(fx_best):
            x_prev = x_best
            fx_prev = fx_best
            x_best = x_new
            fx_best = fx_new
        else:
            x_prev = x_new
            fx_prev = fx_new
            
    return x_best

# EVOLVE-BLOCK-START
# The function is defined above, but the task requires the code to be inside the block or replacing the block content.
# The prompt says "Keep the function names and signatures of the seed program." and "Output the COMPLETE program".
# The seed program has the function inside EVOLVE-BLOCK. I will place the whole logic there.
# EVOLVE-BLOCK-END
