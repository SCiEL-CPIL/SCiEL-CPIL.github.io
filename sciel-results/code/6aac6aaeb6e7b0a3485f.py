"""Root finding: Ill-conditioned - Brent's method with adaptive bracketing."""
import math
import time
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Primary strategy: Brent's method (scipy.optimize.brentq) with adaptive
    bracket expansion to find a sign change. Falls back to best-point tracking.
    """
    start_time = time.time()
    MAX_WALL = 115.0  # Safety margin before 120s limit
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    eval_count = [0]
    
    def eval_f(x):
        """Counted evaluation of f."""
        eval_count[0] += 1
        return f(x)
    
    def time_ok():
        return time.time() - start_time < MAX_WALL
    
    def is_good(x, fx):
        return math.isfinite(fx) and abs(fx) <= TARGET_TOL
    
    x0 = float(x0)
    fx0 = eval_f(x0)
    
    best_x = x0
    best_fx = fx0
    
    if is_good(x0, fx0):
        return x0
    
    if not math.isfinite(fx0):
        # If initial point is bad, try a few nearby points to get a finite value
        for h in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            if not time_ok() or eval_count[0] >= MAX_EVALS:
                break
            x_try = x0 + h
            fx_try = eval_f(x_try)
            if math.isfinite(fx_try):
                x0 = x_try
                fx0 = fx_try
                if is_good(x0, fx0):
                    return x0
                break
    
    if not math.isfinite(fx0):
        return best_x

    # Strategy: Find a bracket [a, b] such that f(a) * f(b) < 0
    # Then use brentq to find the root.
    
    a, b = None, None
    fa, fb = None, None
    
    # We know f(x0) = fx0.
    # Expand outwards from x0 to find a sign change.
    
    # Initial step size
    # Use a scale based on x0 to handle large/small x
    scale = max(1.0, abs(x0))
    
    # Try a sequence of expanding brackets
    # We check both sides.
    # To be efficient, we alternate or check both at each scale.
    
    # Let's try a set of scales
    scales = [
        1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1e4, 1e5, 1e6, 1e8
    ]
    
    found_bracket = False
    
    for s in scales:
        if not time_ok() or eval_count[0] >= MAX_EVALS - 5:
            break
            
        h = s * scale
        
        # Check right side: x0 + h
        x_r = x0 + h
        if x_r != x0:
            fx_r = eval_f(x_r)
            if math.isfinite(fx_r):
                if abs(fx_r) < abs(best_fx):
                    best_x, best_fx = x_r, fx_r
                if is_good(x_r, fx_r):
                    return x_r
                
                if fx0 * fx_r < 0:
                    a, b = (x0, x_r) if x0 < x_r else (x_r, x0)
                    fa, fb = (fx0, fx_r) if x0 < x_r else (fx_r, fx0)
                    found_bracket = True
                    break
        
        # Check left side: x0 - h
        x_l = x0 - h
        if x_l != x0:
            fx_l = eval_f(x_l)
            if math.isfinite(fx_l):
                if abs(fx_l) < abs(best_fx):
                    best_x, best_fx = x_l, fx_l
                if is_good(x_l, fx_l):
                    return x_l
                
                if fx0 * fx_l < 0:
                    a, b = (x0, x_l) if x0 < x_l else (x_l, x0)
                    fa, fb = (fx0, fx_l) if x0 < x_l else (fx_l, fx0)
                    found_bracket = True
                    break
    
    if found_bracket:
        # Use Brent's method
        # We have used some evaluations. brentq uses its own counting? No, we pass f.
        # But we need to count those evaluations.
        # brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
        # We need to ensure we don't exceed MAX_EVALS.
        # We can't easily pass a counter to brentq without wrapping f.
        
        def f_brent(x):
            if eval_count[0] >= MAX_EVALS:
                # Stop early? brentq might not handle this gracefully.
                # Better to check before calling and limit maxiter.
                pass
            return eval_f(x)
        
        # Estimate remaining budget
        remaining = MAX_EVALS - eval_count[0]
        if remaining > 5:
            try:
                # brentq is robust. xtol/rtol set to be well below 1e-8.
                # maxiter should be limited to ensure we don't blow the eval budget.
                # Typically brentq converges in ~10-20 iters.
                root = brentq(f_brent, a, b, xtol=1e-12, rtol=1e-12, maxiter=min(50, remaining))
                if math.isfinite(root):
                    fx_root = eval_f(root) # One more eval to verify? Or trust brentq?
                    # brentq guarantees f(root) is small. But let's check.
                    if is_good(root, fx_root):
                        return root
                    # If not good (rare), maybe it stopped due to maxiter?
                    # Return best_x? Or root if it's better?
                    if abs(fx_root) < abs(best_fx):
                        best_x, best_fx = root, fx_root
            except Exception:
                # If brentq fails, fall through
                pass
    
    # If no bracket found, or brentq failed:
    # Fallback: Use Secant/Newton from best_x?
    # The current program's fallback was Newton/Secant.
    # But since we are pivoting to Brent, maybe a simple local search from best_x is better?
    # Or just return best_x.
    
    # Let's try a few local secant steps from best_x if we have budget.
    if time_ok() and eval_count[0] < MAX_EVALS - 10:
        # Secant from best_x and a nearby point
        h_sec = 1e-4 * max(1.0, abs(best_x))
        x2 = best_x + h_sec
        if x2 != best_x:
            fx2 = eval_f(x2)
            if is_good(x2, fx2):
                return x2
            
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
                x2, fx2 = x_new, eval_f(x_new)
                
                if is_good(x1, fx1): return x1
                if is_good(x2, fx2): return x2
                
                if abs(fx1) < abs(best_fx):
                    best_x, best_fx = x1, fx1
                if abs(fx2) < abs(best_fx):
                    best_x, best_fx = x2, fx2

    return best_x

# EVOLVE-BLOCK-END
