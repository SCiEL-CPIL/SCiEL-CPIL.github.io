"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin for 120s limit
    
    def time_ok():
        return (time.time() - start_time) < deadline
    
    x = float(x0)
    if not math.isfinite(x):
        x = 0.0
    
    # Evaluate initial point
    fx = f(x)
    if not math.isfinite(fx):
        # Try some other points
        for dx in [1.0, -1.0, 0.1, -0.1, 0.01, -0.01]:
            x_try = x + dx
            try:
                fx_try = f(x_try)
                if math.isfinite(fx_try) and abs(fx_try) < abs(fx):
                    x, fx = x_try, fx_try
                    break
            except:
                continue
        if not math.isfinite(fx):
            return x
    
    if abs(fx) <= 1e-8:
        return x
    
    # Phase 1: Bracket finding
    # Try to find an interval [a, b] where f(a) and f(b) have opposite signs
    a, b = None, None
    fa, fb = None, None
    
    # First, try to find a sign change near x0
    step_sizes = [0.1, 0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    for step in step_sizes:
        if not time_ok():
            break
        for sign in [1, -1]:
            if not time_ok():
                break
            b_candidate = x + sign * step
            try:
                fb_candidate = f(b_candidate)
            except:
                continue
            if not math.isfinite(fb_candidate):
                continue
            if fx * fb_candidate < 0:
                a, fa = x, fx
                b, fb = b_candidate, fb_candidate
                break
        if a is not None:
            break
    
    # If no bracket found, try a denser search
    if a is None:
        # Try points around x0
        candidates = []
        base_step = 0.01
        for i in range(-50, 51):
            if not time_ok():
                break
            xi = x + i * base_step
            try:
                fxi = f(xi)
            except:
                continue
            if not math.isfinite(fxi):
                continue
            candidates.append((xi, fxi))
        
        # Look for sign changes in candidates
        for i in range(len(candidates) - 1):
            if not time_ok():
                break
            xi, fxi = candidates[i]
            xj, fxj = candidates[i + 1]
            if fxi * fxj < 0:
                a, fa = xi, fxi
                b, fb = xj, fxj
                break
        
        # If still no bracket, try larger range
        if a is None:
            base_step = 0.1
            for i in range(-200, 201):
                if not time_ok():
                    break
                xi = x + i * base_step
                try:
                    fxi = f(xi)
                except:
                    continue
                if not math.isfinite(fxi):
                    continue
                if abs(fxi) < 1e-8:
                    return xi
                if a is None:
                    a, fa = xi, fxi
                else:
                    if fa * fxi < 0:
                        b, fb = xi, fxi
                        break
    
    # If we have a bracket, use Illinois method (modified bisection)
    if a is not None and b is not None:
        # Ensure |fa| <= |fb| for Illinois method
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        # Illinois method: faster than bisection, more robust than Newton
        max_iter = 200
        for _ in range(max_iter):
            if not time_ok():
                break
            if abs(fb) < abs(fa):
                a, b = b, a
                fa, fb = fb, fa
            
            if abs(fb) <= 1e-8:
                return b
            if abs(fa) <= 1e-8:
                return a
            
            # Illinois: replace the larger magnitude endpoint with midpoint
            # but shrink the interval more aggressively
            m = 0.5 * (a + b)
            try:
                fm = f(m)
            except:
                break
            if not math.isfinite(fm):
                break
            
            if abs(fm) <= 1e-8:
                return m
            
            if fa * fm < 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
                # Illinois: shrink a side
                a = 0.5 * (a + b)
                try:
                    fa = f(a)
                except:
                    break
                if not math.isfinite(fa):
                    break
            
            # Check convergence
            if abs(b - a) < 1e-12:
                return 0.5 * (a + b)
        
        # Return the point with smaller |f|
        if abs(fa) <= abs(fb):
            return a
        else:
            return b
    
    # If no bracket found, use a local minimization approach
    # Try to minimize |f(x)| using a simple search
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)
    
    # Try a few different step sizes
    for step in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
        if not time_ok():
            break
        for direction in [1, -1]:
            if not time_ok():
                break
            x_try = best_x + direction * step
            try:
                fx_try = f(x_try)
            except:
                continue
            if not math.isfinite(fx_try):
                continue
            if abs(fx_try) < best_abs_fx:
                best_x = x_try
                best_fx = fx_try
                best_abs_fx = abs(fx_try)
    
    return best_x
# EVOLVE-BLOCK-END
