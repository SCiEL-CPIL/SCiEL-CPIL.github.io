import math
import time
import numpy as np

def find_root(f, x0):
    tol = 1e-8
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    x = float(x0)
    fx = f(x)
    if abs(fx) <= tol:
        return x
    
    # Try to find a bracket [a, b] where f(a) * f(b) < 0
    a = x
    fa = fx
    
    bracket_found = False
    b = None
    fb = None
    
    # Geometric expansion: check x +/- step for increasing step sizes
    # Start small, grow by factor of 2
    step = 1e-6
    for i in range(60):
        if time.time() > deadline:
            return x
        
        # Check x + step
        x_plus = x + step
        f_plus = f(x_plus)
        if abs(f_plus) <= tol:
            return x_plus
        if fa * f_plus < 0:
            b = x_plus
            fb = f_plus
            bracket_found = True
            break
        
        # Check x - step
        x_minus = x - step
        f_minus = f(x_minus)
        if abs(f_minus) <= tol:
            return x_minus
        if fa * f_minus < 0:
            b = x_minus
            fb = f_minus
            bracket_found = True
            break
        
        step *= 2.0
    
    if not bracket_found:
        # No bracket found; try a dense grid scan over a reasonable range
        if time.time() < deadline - 10.0:
            # Adaptive grid: start with moderate range, expand if needed
            for range_mult in [10.0, 100.0, 1000.0, 10000.0]:
                if time.time() > deadline:
                    return x
                
                grid_start = x - range_mult
                grid_end = x + range_mult
                n_points = 2000
                grid = np.linspace(grid_start, grid_end, n_points)
                
                try:
                    f_vals = np.array([f(xi) for xi in grid])
                except:
                    f_vals = None
                
                if f_vals is not None and np.all(np.isfinite(f_vals)):
                    signs = np.sign(f_vals)
                    # Handle zeros
                    zero_idx = np.where(signs == 0)[0]
                    if len(zero_idx) > 0:
                        idx = zero_idx[0]
                        a = grid[idx]
                        fa = 0.0
                        b = grid[idx + 1] if idx + 1 < len(grid) else grid[idx]
                        fb = f_vals[idx + 1] if idx + 1 < len(grid) else 0.0
                        bracket_found = True
                        break
                    
                    sign_changes = np.where(signs[:-1] * signs[1:] < 0)[0]
                    if len(sign_changes) > 0:
                        idx = sign_changes[0]
                        a = grid[idx]
                        b = grid[idx + 1]
                        fa = f_vals[idx]
                        fb = f_vals[idx + 1]
                        bracket_found = True
                        break
        
        if not bracket_found:
            return x
    
    # Hybrid Secant/Bisection on [a, b]
    lo = a
    hi = b
    flo = fa
    fhi = fb
    
    # Ensure lo < hi
    if lo > hi:
        lo, hi = hi, lo
        flo, fhi = fhi, flo
    
    # Keep track of last two points for secant method
    x_prev1 = lo
    f_prev1 = flo
    x_prev2 = hi
    f_prev2 = fhi
    
    # Hybrid Solver: Use secant when possible, bisection as fallback
    for i in range(80):
        if time.time() > deadline:
            return (lo + hi) / 2.0
            
        if abs(hi - lo) < 1e-13:
            break
            
        # Try secant step using the two most recent points
        denom = f_prev2 - f_prev1
        x_new = None
        
        if abs(denom) > 1e-300:
            x_secant = x_prev2 - f_prev2 * (x_prev2 - x_prev1) / denom
            # Only use secant if it's within the bracket (or close)
            bracket_mid = (lo + hi) / 2.0
            bracket_half = (hi - lo) / 2.0
            if abs(x_secant - bracket_mid) < bracket_half * 1.5:
                x_new = x_secant
        
        # Fallback to bisection
        if x_new is None:
            x_new = (lo + hi) / 2.0
        
        f_new = f(x_new)
        
        if abs(f_new) <= tol:
            return x_new
        
        # Update bracket
        if flo * f_new < 0:
            hi = x_new
            fhi = f_new
        else:
            lo = x_new
            flo = f_new
        
        # Update last two points for secant
        x_prev1 = x_prev2
        f_prev1 = f_prev2
        x_prev2 = x_new
        f_prev2 = f_new
        
        # Early exit if bracket is very small
        if abs(hi - lo) < 1e-12:
            break
    
    # Final polish: use the midpoint of the tightest bracket
    return (lo + hi) / 2.0
