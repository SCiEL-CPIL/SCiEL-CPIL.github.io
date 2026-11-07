"""Root finder for 3 fixed oscillatory equations."""
import time
import math
import numpy as np
from scipy.optimize import brentq

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 for an oscillatory equation.
    Strategy:
      1. Try Newton's method with a few steps (fast if converging).
      2. If Newton fails, do a local bracket search around x0.
      3. If bracket found, use Brent's method (scipy.optimize.brentq).
      4. If no bracket, expand search range and try again.
      5. Deadline guard: return best candidate found if time runs out.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    x = float(x0)
    best_x = x
    best_fx = None
    try:
        best_fx = f(x)
        if math.isfinite(best_fx):
            pass
        else:
            best_fx = float('inf')
    except Exception:
        best_fx = float('inf')
    
    def time_check():
        return (time.time() - start_time) > deadline
    
    # Phase 1: Newton's method with forward difference
    for i in range(20):
        if time_check():
            break
        try:
            fx = f(x)
        except Exception:
            break
        if not math.isfinite(fx):
            break
        if abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx
        if abs(fx) <= 1e-10:
            return x
        
        h = 1e-7 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
        except Exception:
            break
        if not math.isfinite(fx_plus):
            break
        
        d = (fx_plus - fx) / h
        if abs(d) < 1e-15:
            break
        
        new_x = x - fx / d
        # Damping if step is too large
        if abs(new_x - x) > 10.0 * max(1.0, abs(x)):
            new_x = x + 0.5 * max(1.0, abs(x)) * math.copysign(1, -fx / d)
        x = new_x
    
    # Phase 2: Bracket search and Brent's method
    # Try to find a bracket [a, b] where f(a) and f(b) have opposite signs
    def try_brent(a, b):
        """Try Brent's method on bracket [a, b]."""
        if time_check():
            return None
        try:
            fa = f(a)
            fb = f(b)
            if not (math.isfinite(fa) and math.isfinite(fb)):
                return None
            if fa * fb > 0:
                return None
            # Use brentq with maxiter to limit calls
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if math.isfinite(root):
                fr = f(root)
                if math.isfinite(fr) and abs(fr) < abs(best_fx):
                    best_x = root
                    best_fx = fr
                if abs(fr) <= 1e-8:
                    return root
            return None
        except Exception:
            return None
    
    # Start with a small bracket around x0
    step = 0.1 * max(1.0, abs(x0))
    found = None
    
    for expansion in range(20):
        if time_check():
            break
        a = x0 - (expansion + 1) * step
        b = x0 + (expansion + 1) * step
        
        # Try several sub-intervals within [a, b] to find sign changes
        n_sub = 20
        for j in range(n_sub):
            if time_check():
                break
            xa = a + j * (b - a) / n_sub
            xb = a + (j + 1) * (b - a) / n_sub
            try:
                fa = f(xa)
                fb = f(xb)
            except Exception:
                continue
            if not (math.isfinite(fa) and math.isfinite(fb)):
                continue
            if abs(fa) < abs(best_fx):
                best_x = xa
                best_fx = fa
            if abs(fb) < abs(best_fx):
                best_x = xb
                best_fx = fb
            if fa * fb < 0:
                found = try_brent(xa, xb)
                if found is not None:
                    return found
    
    # If we found a root via Brent, return it
    if found is not None:
        return found
    
    # Phase 3: If no bracket found, try a denser scan over a wider range
    if not time_check():
        for scale in [1.0, 5.0, 10.0, 50.0, 100.0]:
            if time_check():
                break
            a = x0 - scale * max(1.0, abs(x0))
            b = x0 + scale * max(1.0, abs(x0))
            n_points = 100
            prev_x = None
            prev_f = None
            for i in range(n_points + 1):
                if time_check():
                    break
                xi = a + i * (b - a) / n_points
                try:
                    fi = f(xi)
                except Exception:
                    prev_x = None
                    prev_f = None
                    continue
                if not math.isfinite(fi):
                    prev_x = None
                    prev_f = None
                    continue
                if abs(fi) < abs(best_fx):
                    best_x = xi
                    best_fx = fi
                if prev_x is not None and prev_f is not None:
                    if prev_f * fi < 0:
                        found = try_brent(prev_x, xi)
                        if found is not None:
                            return found
                prev_x = xi
                prev_f = fi
    
    # Return best candidate found
    return best_x
# EVOLVE-BLOCK-END
