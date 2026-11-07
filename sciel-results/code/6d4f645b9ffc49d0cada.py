"""
Root finder using a model-and-optimize approach with adaptive sampling.
Uses polynomial fitting on sampled points to model f, then optimizes
the model to find the root. Falls back to bisection for robustness.
"""
import time
import math
import numpy as np

def find_root(f, x0):
    """
    Finds a root x of f(x) such that |f(x)| <= 1e-8.
    Uses adaptive model-based approach for smooth polynomials.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    TOL = 1e-9
    MAX_CALLS = 480
    
    x = float(x0)
    call_count = 0
    cache = {}
    
    def _elapsed():
        return time.time() - start_time
    
    def _time_ok():
        return _elapsed() < deadline and call_count < MAX_CALLS
    
    def _f(x_val):
        """Cached function evaluation with call counting."""
        nonlocal call_count
        if call_count >= MAX_CALLS:
            return None
        key = x_val
        if key in cache:
            return cache[key]
        try:
            val = f(key)
            call_count += 1
            if not math.isfinite(val):
                cache[key] = None
                return None
            cache[key] = val
            return val
        except Exception:
            call_count += 1
            return None

    # Phase 0: Check if x0 is already a root
    fx0 = _f(x)
    if fx0 is None:
        return x
    if abs(fx0) <= TOL:
        return x
    
    # Phase 1: Adaptive sampling to build a model
    # Start with a small set of points, expand adaptively
    # We want to find a bracket [a, b] where f(a)*f(b) < 0
    
    a = b = None
    fa = fb = None
    
    # Initial sampling: exponential expansion from x0
    step = 1e-3
    max_expansions = 30
    points = [(x, fx0)]
    
    for i in range(max_expansions):
        if not _time_ok():
            break
        
        # Try right
        xr = x + step
        fr = _f(xr)
        if fr is not None:
            points.append((xr, fr))
            if abs(fr) <= TOL:
                return xr
            if fx0 is not None and fx0 * fr < 0:
                a, b = x, xr
                fa, fb = fx0, fr
                break
        
        # Try left
        xl = x - step
        fl = _f(xl)
        if fl is not None:
            points.append((xl, fl))
            if abs(fl) <= TOL:
                return xl
            if fx0 is not None and fx0 * fl < 0:
                a, b = xl, x
                fa, fb = fl, fx0
                break
        
        step *= 2.0
    
    # If no bracket found, use grid search
    if a is None or b is None:
        best_x = x
        best_abs = abs(fx0)
        
        max_range = max(abs(p[0] - x) for p in points) if points else 1.0
        search_range = max_range * 10 + 10.0
        
        n_samples = 50
        prev_x = None
        prev_f = None
        for i in range(n_samples + 1):
            if not _time_ok():
                break
            t = i / n_samples
            x_try = x - search_range + 2 * search_range * t
            f_try = _f(x_try)
            if f_try is not None:
                if abs(f_try) < best_abs:
                    best_abs = abs(f_try)
                    best_x = x_try
                if abs(f_try) <= TOL:
                    return x_try
                if prev_f is not None and prev_f * f_try < 0:
                    a, b = prev_x, x_try
                    fa, fb = prev_f, f_try
                    break
                prev_x = x_try
                prev_f = f_try
        
        if a is not None and b is not None:
            # Found bracket via grid search
            pass
        else:
            # No bracket, use secant method from best_x
            x_curr = best_x
            fx_curr = _f(x_curr)
            if fx_curr is None:
                return best_x
            
            # Secant method
            x_prev = x_curr + 1e-4 * max(1.0, abs(x_curr))
            fx_prev = _f(x_prev)
            if fx_prev is None:
                return best_x
            
            for _ in range(50):
                if not _time_ok():
                    break
                if abs(fx_curr) <= TOL:
                    return x_curr
                if abs(fx_prev) <= TOL:
                    return x_prev
                
                denom = fx_curr - fx_prev
                if abs(denom) < 1e-15:
                    break
                x_new = x_curr - fx_curr * (x_curr - x_prev) / denom
                if not math.isfinite(x_new):
                    break
                
                x_prev, fx_prev = x_curr, fx_curr
                x_curr = x_new
                fx_curr = _f(x_curr)
                if fx_curr is None:
                    break
            
            return x_curr

    # Phase 2: Model-based refinement
    # We have a bracket [a, b] with fa*fb < 0
    # Fit a polynomial model to the points and find its root
    
    x_lo, x_hi = a, b
    f_lo, f_hi = fa, fb
    
    # Collect all points in [a, b] plus endpoints
    bracket_points = [(x_lo, f_lo), (x_hi, f_hi)]
    for px, pf in points:
        if x_lo <= px <= x_hi:
            bracket_points.append((px, pf))
    
    # Remove duplicates
    seen = set()
    unique_points = []
    for px, pf in bracket_points:
        key = px
        if key not in seen:
            seen.add(key)
            unique_points.append((px, pf))
    
    # Fit polynomial model using least squares
    # Start with degree 2, increase if needed
    x_data = np.array([p[0] for p in unique_points])
    f_data = np.array([p[1] for p in unique_points])
    
    # Find root of the fitted polynomial
    def _find_poly_root(coeffs, x_lo, x_hi):
        """Find root of polynomial defined by coeffs in [x_lo, x_hi]."""
        # Use numpy roots
        try:
            roots = np.roots(coeffs)
            # Filter real roots in [x_lo, x_hi]
            real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and x_lo <= r.real <= x_hi]
            if real_roots:
                # Choose the root that minimizes |f(x)| in the bracket
                best_root = min(real_roots, key=lambda r: abs(f(r)))
                return best_root
        except Exception:
            pass
        return None
    
    # Try increasing degrees
    for degree in [2, 3, 4, 5]:
        if len(unique_points) < degree + 1:
            break
        
        # Fit polynomial of given degree
        try:
            coeffs = np.polyfit(x_data, f_data, degree)
            root = _find_poly_root(coeffs, x_lo, x_hi)
            if root is not None:
                # Evaluate f at the root
                f_root = _f(root)
                if f_root is not None:
                    if abs(f_root) <= TOL:
                        return root
                    # Update bracket
                    if f_root * f_lo < 0:
                        x_hi, f_hi = root, f_root
                    else:
                        x_lo, f_lo = root, f_root
                    # Add point to our data
                    unique_points.append((root, f_root))
                    x_data = np.array([p[0] for p in unique_points])
                    f_data = np.array([p[1] for p in unique_points])
        except Exception:
            continue
    
    # Phase 3: Bisection fallback for final convergence
    # Use bisection to guarantee convergence within the bracket
    for _ in range(100):
        if not _time_ok():
            break
        
        if abs(f_lo) <= TOL:
            return x_lo
        if abs(f_hi) <= TOL:
            return x_hi
        
        x_mid = (x_lo + x_hi) / 2.0
        f_mid = _f(x_mid)
        if f_mid is None:
            break
        
        if abs(f_mid) <= TOL:
            return x_mid
        
        if f_mid * f_lo < 0:
            x_hi, f_hi = x_mid, f_mid
        else:
            x_lo, f_lo = x_mid, f_mid
    
    # Return the best estimate
    if abs(f_lo) < abs(f_hi):
        return x_lo
    else:
        return x_hi
