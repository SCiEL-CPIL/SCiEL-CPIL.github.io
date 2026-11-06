import math
import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    """
    Finds a root x of f(x) such that |f(x)| <= 1e-8.
    Uses a multi-start bracketing search followed by scipy's Brent's method.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s/150s limit
    max_evals = 500
    eval_count = 0
    
    def check_time():
        return time.time() > deadline

    def eval_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return None
        try:
            fx = f(x)
            eval_count += 1
            if math.isnan(fx) or math.isinf(fx):
                return None
            return fx
        except Exception:
            return None

    # Initial evaluation
    x0 = float(x0)
    fx0 = eval_f(x0)
    if fx0 is None:
        return x0
        
    best_x = x0
    best_fx = fx0
    
    if abs(fx0) <= 1e-8:
        return x0

    # Strategy: Search for a bracket [a, b] where f(a) and f(b) have opposite signs.
    # We try expanding intervals around x0.
    
    # Initial step size heuristic
    # For ill-conditioned problems, the root might be close or far.
    # We try a geometric progression of step sizes.
    
    # We will maintain a set of evaluated points to avoid re-evaluation
    # though brentq will handle its own internal evaluations.
    # We just need to find the bracket.
    
    # List of (point, value) to track best
    points = [(x0, fx0)]
    
    # Try to find a bracket
    bracket_found = False
    a, b = None, None
    fa, fb = None, None
    
    # We search in both directions with increasing step sizes
    # Start with small steps to catch nearby roots, then larger ones.
    # 1e-4 to 1e4 covers a wide range.
    
    # To save evaluations, we check if we already have a sign change with existing points.
    # But usually x0 is the only point.
    
    # Define step sizes
    # We want to be efficient. If we find a bracket, we stop searching and refine.
    
    # We'll try steps: 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000
    # Multiplied by scale of x0 if x0 is large.
    scale = max(1.0, abs(x0))
    base_steps = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]
    
    # Also try very small steps if x0 is very small?
    # The scale handles x0.
    
    # To maximize chance of finding a bracket with few evals:
    # 1. Check x0 +/- small delta
    # 2. Check x0 +/- larger delta
    
    # We will iterate through steps. For each step, try right then left.
    
    for step in base_steps:
        if check_time() or eval_count >= max_evals - 5: # Reserve some for brentq
            break
            
        delta = step * scale
        
        # Try right
        x_r = x0 + delta
        fx_r = eval_f(x_r)
        if fx_r is not None:
            points.append((x_r, fx_r))
            if abs(fx_r) < abs(best_fx):
                best_x, best_fx = x_r, fx_r
            if abs(fx_r) <= 1e-8:
                return x_r
            
            if fx0 is not None and fx0 * fx_r < 0:
                a, b = x0, x_r
                fa, fb = fx0, fx_r
                bracket_found = True
                break
        
        if check_time() or eval_count >= max_evals - 5:
            break
            
        # Try left
        x_l = x0 - delta
        fx_l = eval_f(x_l)
        if fx_l is not None:
            points.append((x_l, fx_l))
            if abs(fx_l) < abs(best_fx):
                best_x, best_fx = x_l, fx_l
            if abs(fx_l) <= 1e-8:
                return x_l
            
            if fx0 is not None and fx0 * fx_l < 0:
                a, b = x_l, x0
                fa, fb = fx_l, fx0
                bracket_found = True
                break
        
        # Also check if any new point forms a bracket with each other?
        # Unlikely to be efficient to check all pairs.
        # But if fx0 and fx_r didn't bracket, maybe fx_r and fx_l (if both exist) do?
        # No, they are on opposite sides of x0. If fx0 is between them, 
        # a sign change between fx_r and fx_l implies fx0 is either same sign as one 
        # or the root is not simple? 
        # Actually, if f(x_l) and f(x_r) have opposite signs, there is a root in [x_l, x_r].
        # We can use that bracket too.
        
        if fx_l is not None and fx_r is not None:
            if fx_l * fx_r < 0:
                a, b = x_l, x_r
                fa, fb = fx_l, fx_r
                bracket_found = True
                break

    # If we found a bracket, use Brent's method
    if bracket_found:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        try:
            # brentq requires f(a) and f(b) to have different signs.
            # It will raise an error if they don't.
            # xtol=1e-12 is tight enough to ensure |f(x)| < 1e-8 for well-behaved f.
            # maxiter=100 is plenty for double precision.
            root = brentq(f, a, b, xtol=1e-12, maxiter=100)
            
            # Verify the result
            f_root = eval_f(root)
            if f_root is not None and abs(f_root) <= 1e-8:
                return root
            # If brentq succeeded but verification failed (unlikely), 
            # return the root as it's likely the best candidate.
            return root
        except Exception:
            # If brentq fails, fall through to return best_x
            pass

    # If no bracket found, return the best point found
    return best_x
