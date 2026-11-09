"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize

def _safe_f(f, x):
    try:
        y = f(x)
        if not math.isfinite(y):
            return float('inf')
        return float(y)
    except Exception:
        return float('inf')

def _residual_func(f, x_vec):
    """Objective function for minimization: square of f(x) to handle signs."""
    x = float(x_vec[0])
    fx = _safe_f(f, x)
    if not math.isfinite(fx):
        return 1e100
    return fx * fx

def _newton_refine(f, x, max_iter=10):
    """
    Fast local refinement using Newton's method with finite difference derivative.
    Used only when we are close enough to a root for Newton to be stable.
    """
    x = float(x)
    for _ in range(max_iter):
        fx = _safe_f(f, x)
        if not math.isfinite(fx):
            break
        if abs(fx) <= 1e-12:
            return x
        
        h = 1e-6 * max(1.0, abs(x))
        fx_plus = _safe_f(f, x + h)
        fx_minus = _safe_f(f, x - h)
        
        if not math.isfinite(fx_plus) or not math.isfinite(fx_minus):
            break
            
        # Central difference for better derivative accuracy
        dfx = (fx_plus - fx_minus) / (2 * h)
        
        if abs(dfx) < 1e-15:
            break
            
        x_new = x - fx / dfx
        
        if not math.isfinite(x_new):
            break
            
        # Check for divergence or oscillation
        if abs(x_new - x) > 10 * abs(h) * 100:
            break
            
        x = x_new
        
    return x

def find_root(f, x0):
    """
    Global root finder using Nelder-Mead minimization of |f(x)|^2.
    1. Starts with a simplex centered at x0 with adaptive initial size.
    2. Runs Nelder-Mead to find a local minimum of the residual.
    3. Refines the result with Newton's method if close enough.
    4. Falls back to bisection if a sign change is detected during the process (optional, but NM is usually sufficient for polynomials).
    """
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    x0 = float(x0)
    if not math.isfinite(x0):
        x0 = 0.0
        
    # Initial check
    fx0 = _safe_f(f, x0)
    if math.isfinite(fx0) and abs(fx0) <= 1e-8:
        return x0

    # Prepare for Nelder-Mead
    # Initial simplex size: start with a small perturbation, but NM will expand if needed.
    # However, if the root is far, a small initial size might get stuck in a local min of |f|^2 
    # that isn't a root (though for polynomials, local mins of |f|^2 are often roots or close).
    # To be safe against "flat" regions or distant roots, we can try a slightly larger initial step.
    # Let's use an initial step of 1.0 or 10.0 depending on x0 magnitude?
    # Standard NM often works well with default settings if x0 is decent.
    # Let's try a few different initial simplex sizes if the first one fails?
    # No, keep it simple. Use a reasonable initial size.
    
    # If x0 is 0, start with step 1.0. If x0 is large, step 1.0 might be too small to see curvature.
    # Let's use step = max(1.0, abs(x0) * 0.1)
    initial_step = max(1.0, abs(x0) * 0.1)
    
    x_initial = np.array([x0])
    
    # Nelder-Mead options
    # maxiter: 500 is usually plenty for 1D.
    # xatol, fatol: tight tolerances.
    options = {
        'maxiter': 500,
        'xatol': 1e-12,
        'fatol': 1e-16,
        'adaptive': True
    }
    
    best_x = x0
    best_fx = abs(fx0) if math.isfinite(fx0) else float('inf')
    
    # Try Nelder-Mead
    try:
        if time.time() < deadline - 5.0:
            res = minimize(
                _residual_func, 
                x_initial, 
                args=(f,), 
                method='Nelder-Mead', 
                options=options
            )
            
            x_nm = float(res.x[0])
            fx_nm = _safe_f(f, x_nm)
            
            if math.isfinite(fx_nm):
                if abs(fx_nm) < best_fx:
                    best_x = x_nm
                    best_fx = abs(fx_nm)
                # If NM found a very small residual, we are likely at a root
                if abs(fx_nm) <= 1e-8:
                    # Refine with Newton for extra precision (optional but good)
                    x_refined = _newton_refine(f, x_nm, max_iter=10)
                    fx_refined = _safe_f(f, x_refined)
                    if math.isfinite(fx_refined) and abs(fx_refined) <= 1e-8:
                        return x_refined
                    return x_nm
    except Exception:
        pass

    # If NM didn't get us close enough, try a different initial step size?
    # Sometimes if x0 is far from the root and the function is flat, NM might stall.
    # Let's try a larger step if we didn't succeed.
    if best_fx > 1e-8 and time.time() < deadline - 5.0:
        # Try with a larger initial step
        initial_step2 = initial_step * 10.0
        x_initial2 = np.array([x0 + initial_step2])
        try:
            res2 = minimize(
                _residual_func, 
                x_initial2, 
                args=(f,), 
                method='Nelder-Mead', 
                options=options
            )
            x_nm2 = float(res2.x[0])
            fx_nm2 = _safe_f(f, x_nm2)
            if math.isfinite(fx_nm2) and abs(fx_nm2) < best_fx:
                best_x = x_nm2
                best_fx = abs(fx_nm2)
                if abs(fx_nm2) <= 1e-8:
                    x_refined = _newton_refine(f, x_nm2, max_iter=10)
                    fx_refined = _safe_f(f, x_refined)
                    if math.isfinite(fx_refined) and abs(fx_refined) <= 1e-8:
                        return x_refined
                    return x_nm2
        except Exception:
            pass

    # Fallback: If NM failed (e.g. local minimum of |f|^2 that isn't a root, or divergence),
    # try a global bracketing search as a last resort.
    # This is similar to the old Phase 2 but only if NM failed.
    if best_fx > 1e-8 and time.time() < deadline - 10.0:
        a = x0
        fa = _safe_f(f, a)
        if not math.isfinite(fa):
            a = 0.0
            fa = _safe_f(f, a)
            if not math.isfinite(fa):
                return best_x

        if abs(fa) <= 1e-8:
            return a

        scale = 1.0
        found_bracket = False
        # Re-use best_x from NM as a candidate
        while time.time() < deadline - 5.0 and scale < 1e10:
            for direction in [1.0, -1.0]:
                b = x0 + direction * scale
                fb = _safe_f(f, b)
                if not math.isfinite(fb):
                    continue
                
                if fa * fb < 0:
                    found_bracket = True
                    # Refine with Bisection
                    for _ in range(50):
                        if time.time() > deadline - 2.0:
                            break
                        m = (a + b) / 2.0
                        fm = _safe_f(f, m)
                        if not math.isfinite(fm):
                            break
                        if abs(fm) <= 1e-8:
                            return m
                        if fa * fm <= 0:
                            b = m
                            fb = fm
                        else:
                            a = m
                            fa = fm
                        if abs(b - a) < 1e-12:
                            break
                    return (a + b) / 2.0
            
            scale *= 2.0
            
    return best_x

# EVOLVE-BLOCK-END
