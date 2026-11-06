"""Root finding: Ill-conditioned - Global Optimization Strategy."""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Strategy: 
    1. Use Differential Evolution to find a promising region for f(x)^2.
    2. Use L-BFGS-B (Trust Region) to refine the solution.
    3. Fallback to standard methods if optimization fails.
    """
    start_time = time.time()
    MAX_WALL = 110.0  # Safety margin
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    eval_count = [0]
    
    def time_ok():
        return time.time() - start_time < MAX_WALL

    def eval_f(x):
        """Counted evaluation of f."""
        eval_count[0] += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return 1e100
            return val
        except Exception:
            return 1e100

    # Objective function for optimization: minimize f(x)^2
    def objective(x_vec):
        if not time_ok() or eval_count[0] >= MAX_EVALS - 10:
            return 1e100
        x = float(x_vec[0])
        fx = eval_f(x)
        if not math.isfinite(fx):
            return 1e100
        return fx * fx

    # Check initial point
    x0 = float(x0)
    fx0 = eval_f(x0)
    if abs(fx0) <= TARGET_TOL:
        return x0

    best_x = x0
    best_fx = fx0
    best_obj = fx0 * fx0

    # Strategy 1: Differential Evolution (Global Search)
    # We need a reasonable bound. We expand from x0.
    # If x0 is near 0, we search [-10, 10]. If large, we search relative.
    span = max(10.0, abs(x0) * 10.0)
    bounds = [(x0 - span, x0 + span)]
    
    # DE parameters:
    # popsize=5 -> ~5*10=50 initial evals + iterations
    # maxiter=5 -> ~250 evals total roughly
    # We limit maxiter to keep within budget
    try:
        if time_ok() and eval_count[0] < MAX_EVALS - 50:
            # Differential evolution is stochastic, so we set seed for reproducibility if needed, 
            # but here we just want a good search.
            # We use a small population to save evaluations.
            de_result = differential_evolution(
                objective, 
                bounds, 
                maxiter=10, 
                popsize=4, 
                seed=42,
                tol=1e-12,
                init='latinhypercube'
            )
            
            if de_result.success or de_result.fun < best_obj:
                x_de = de_result.x[0]
                fx_de = eval_f(x_de)
                if math.isfinite(fx_de):
                    if abs(fx_de) < abs(best_fx):
                        best_x, best_fx = x_de, fx_de
                        best_obj = fx_de * fx_de
                    if abs(fx_de) <= TARGET_TOL:
                        return x_de
    except Exception:
        pass

    # Strategy 2: Local Refinement with L-BFGS-B (Trust Region / Quasi-Newton)
    # Start from best_x found so far
    if time_ok() and eval_count[0] < MAX_EVALS - 20:
        try:
            # L-BFGS-B is robust and uses gradient approximations efficiently
            # We minimize f(x)^2
            res = minimize(
                objective, 
                [best_x], 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'gtol': 1e-10}
            )
            
            if res.fun < best_obj:
                x_min = res.x[0]
                fx_min = eval_f(x_min)
                if math.isfinite(fx_min):
                    if abs(fx_min) < abs(best_fx):
                        best_x, best_fx = x_min, fx_min
                        best_obj = fx_min * fx_min
                    if abs(fx_min) <= TARGET_TOL:
                        return x_min
        except Exception:
            pass

    # Strategy 3: Fallback to Secant/Newton if optimization didn't get close enough
    # But we are out of "global" budget, so let's try a few standard steps from best_x
    if abs(best_fx) > TARGET_TOL and time_ok() and eval_count[0] < MAX_EVALS - 5:
        # Try Secant from best_x and a nearby point
        h = 1e-4 * max(1.0, abs(best_x))
        x2 = best_x + h
        if x2 != best_x:
            fx2 = eval_f(x2)
            if abs(fx2) <= TARGET_TOL:
                return x2
            if abs(fx2) < abs(best_fx):
                best_x, best_fx = x2, fx2
            
            # Simple Secant loop
            x1, fx1 = best_x, best_fx
            for _ in range(20):
                if not time_ok() or eval_count[0] >= MAX_EVALS:
                    break
                if fx1 == fx2 or x1 == x2:
                    break
                d = (fx2 - fx1) / (x2 - x1)
                if d == 0 or not math.isfinite(d):
                    break
                x_new = x1 - fx1 / d
                if not math.isfinite(x_new):
                    break
                x1, fx1 = x2, fx2
                if eval_count[0] >= MAX_EVALS:
                    break
                x2, fx2 = x_new, eval_f(x_new)
                
                if abs(fx1) < abs(best_fx):
                    best_x, best_fx = x1, fx1
                if abs(fx2) < abs(best_fx):
                    best_x, best_fx = x2, fx2
                    
                if abs(fx1) <= TARGET_TOL:
                    return x1
                if abs(fx2) <= TARGET_TOL:
                    return x2

    return best_x

# EVOLVE-BLOCK-END
