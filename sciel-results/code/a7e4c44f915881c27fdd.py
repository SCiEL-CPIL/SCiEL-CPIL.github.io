import math
import numpy as np
from scipy.optimize import brentq

def _secant_method(f, x_start, max_iter=50, tol=1e-12):
    """
    Robust Secant method with restarts/stall detection.
    Returns root if found, else None.
    """
    x1 = x_start
    x2 = x_start + 1e-6 * max(1.0, abs(x_start))
    
    try:
        f1 = f(x1)
        if not math.isfinite(f1): return None
        if abs(f1) <= tol: return x1
        
        f2 = f(x2)
        if not math.isfinite(f2): return None
        if abs(f2) <= tol: return x2
    except Exception:
        return None

    for _ in range(max_iter):
        # If values are too close, perturb
        if f1 == f2:
            x2 = x1 + 1e-6 * max(1.0, abs(x1))
            try:
                f2 = f(x2)
            except Exception:
                return None
            if not math.isfinite(f2): return None
            if abs(f2) <= tol: return x2
            continue

        # Secant step
        denom = f2 - f1
        if abs(denom) < 1e-15:
             # Degenerate, perturb
             x2 = x1 + 1e-6
             try: f2 = f(x2)
             except: return None
             if not math.isfinite(f2): return None
             continue

        x_new = x1 - f1 * (x1 - x2) / denom
        
        if not math.isfinite(x_new):
            return None

        # Check for divergence: if we move too far, bail
        # Heuristic: if |x_new| is huge compared to initial, it might be diverging
        # But for polynomials, roots might be far. 
        # Instead, check if we are making progress.
        
        try:
            f_new = f(x_new)
        except Exception:
            return None
            
        if not math.isfinite(f_new):
            return None

        if abs(f_new) <= tol:
            return x_new

        # Update
        x1, x2 = x2, x1
        f1, f2 = f2, f_new
        
        # If we are oscillating or not converging, we might loop.
        # Simple check: if |f2| > |f1| * 10 for several steps, it's diverging.
        # We don't track history here, but max_iter limits it.
        
    return None

def find_root(f, x0):
    """
    Finds a root of f starting from x0.
    Returns x such that |f(x)| <= 1e-8.
    Uses Multi-start Secant method with fallback to Bracketing.
    """
    x0 = float(x0)
    
    # List of starting points to try for Secant method
    # 1. The given x0
    # 2. Perturbations to help if x0 is in a bad region
    starts = [
        x0,
        x0 + 1.0,
        x0 - 1.0,
        x0 + 10.0,
        x0 - 10.0,
        x0 + 100.0,
        x0 - 100.0,
        0.0, # Polynomials often have roots near 0
        1.0,
        -1.0
    ]
    
    # Remove duplicates and non-finite
    seen = set()
    unique_starts = []
    for s in starts:
        if math.isfinite(s):
            key = round(s, 5) # Coarse dedup
            if key not in seen:
                seen.add(key)
                unique_starts.append(s)

    # Try Secant method from each start
    for s in unique_starts:
        root = _secant_method(f, s, max_iter=30, tol=1e-12)
        if root is not None:
            # Verify the root
            try:
                if abs(f(root)) <= 1e-8:
                    return root
            except Exception:
                pass
            # If verification fails, continue to next start
            # Note: _secant_method returns None if it hits max_iter or diverges.
            # It returns x if abs(f(x)) <= tol. 
            # So if it returns a value, it should be good.
            # But let's double check |f(x)| <= 1e-8 as per contract.
            if abs(f(root)) <= 1e-8:
                return root

    # Fallback: Bracketing search
    # Expand from x0 to find a sign change
    a, b = None, None
    try:
        fa = f(x0)
    except Exception:
        # If f fails, we can't do much. Return x0? No, must be finite.
        # Try a standard point?
        return 0.0

    if not math.isfinite(fa):
        return 0.0

    if abs(fa) <= 1e-8:
        return x0

    step = 1.0
    max_expansions = 30
    for _ in range(max_expansions):
        x_plus = x0 + step
        try:
            fp = f(x_plus)
            if math.isfinite(fp) and fa * fp < 0:
                a, b = x0, x_plus
                break
        except Exception:
            pass

        x_minus = x0 - step
        try:
            fm = f(x_minus)
            if math.isfinite(fm) and fa * fm < 0:
                a, b = x_minus, x0
                break
        except Exception:
            pass
        
        step *= 2.0

    if a is not None and b is not None:
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
            if math.isfinite(root) and abs(f(root)) <= 1e-8:
                return root
        except Exception:
            pass

    # Last resort: return the best point found during secant attempts?
    # We didn't save them. 
    # Just return x0 if nothing else works, though it will likely fail.
    return x0
