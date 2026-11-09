"""Root finder for oscillatory equations using Brent's method with adaptive bracketing."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Uses a combination of:
      1. Quick local check
      2. Adaptive bracketing to find a sign change
      3. Brent's method for robust convergence
      4. Newton refinement for final polish
    """
    x0 = float(x0)
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    def time_ok():
        return time.time() < deadline
    
    # First, check if x0 itself is close to a root
    fx0 = f(x0)
    if abs(fx0) <= 1e-8:
        return x0
    if not math.isfinite(fx0):
        # Try a small perturbation
        x0 += 1e-6
        fx0 = f(x0)
        if not math.isfinite(fx0) or abs(fx0) > 1e8:
            return x0  # Return best we have
    
    # Try to find a bracket [a, b] where f(a) and f(b) have opposite signs
    # Start with a small step and expand
    step = 1e-3
    max_step = 1000.0
    a = x0
    fa = fx0
    
    # Try expanding in both directions
    best_x = x0
    best_abs_f = abs(fx0)
    
    # First, try a few small steps to see if we can find a sign change quickly
    for i in range(20):
        if not time_ok():
            break
        step = 1e-3 * (2 ** i)
        if step > max_step:
            break
        
        # Try x0 + step
        b = x0 + step
        fb = f(b)
        if not math.isfinite(fb):
            continue
        if abs(fb) < best_abs_f:
            best_abs_f = abs(fb)
            best_x = b
        if fa * fb < 0:
            # Found a bracket
            root = _brent(f, a, b, fa, fb)
            if root is not None:
                # Polish with Newton
                root = _newton_polish(f, root)
                return root
            break
        
        # Try x0 - step
        b = x0 - step
        fb = f(b)
        if not math.isfinite(fb):
            continue
        if abs(fb) < best_abs_f:
            best_abs_f = abs(fb)
            best_x = b
        if fa * fb < 0:
            # Found a bracket
            root = _brent(f, b, a, fb, fa)
            if root is not None:
                root = _newton_polish(f, root)
                return root
            break
    
    # If no bracket found, try a finer search around x0
    if time_ok():
        # Sample more densely around x0
        n_samples = 50
        h = 0.1
        xs = [x0 + (i - n_samples // 2) * h / n_samples for i in range(n_samples)]
        fs = []
        for x in xs:
            if not time_ok():
                break
            fx = f(x)
            if not math.isfinite(fx):
                fs.append(None)
            else:
                fs.append(fx)
                if abs(fx) < best_abs_f:
                    best_abs_f = abs(fx)
                    best_x = x
        
        # Look for sign changes in the samples
        for i in range(len(xs) - 1):
            if not time_ok():
                break
            if fs[i] is None or fs[i+1] is None:
                continue
            if fs[i] * fs[i+1] < 0:
                root = _brent(f, xs[i], xs[i+1], fs[i], fs[i+1])
                if root is not None:
                    root = _newton_polish(f, root)
                    return root
    
    # If still no bracket, try Newton from x0
    if time_ok():
        x = x0
        fx = fx0
        for _ in range(50):
            if not time_ok():
                break
            h = 1e-7 * max(1.0, abs(x))
            d = (f(x + h) - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            fx_new = f(x_new)
            if not math.isfinite(fx_new):
                break
            if abs(fx_new) < best_abs_f:
                best_abs_f = abs(fx_new)
                best_x = x_new
            if abs(fx_new) <= 1e-12:
                return x_new
            if abs(x_new - x) < 1e-15 * max(1.0, abs(x)):
                break
            x = x_new
            fx = fx_new
    
    # Return the best point we found
    return best_x


def _brent(f, a, b, fa, fb, max_iter=100):
    """
    Brent's method to find a root in [a, b] where f(a)*f(b) < 0.
    """
    if fa * fb > 0:
        return None
    
    # Ensure |fa| < |fb|
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    
    c = a
    fc = fa
    d = b - a
    e = d
    tol = 1e-14
    
    for _ in range(max_iter):
        # Check for convergence
        if abs(fa) <= 1e-12:
            return a
        if abs(b - a) < tol * max(1.0, abs(a)):
            return a if abs(fa) < abs(fb) else b
        
        # Inverse quadratic interpolation
        if fc != fa and fc != fb:
            # S = fb
            s = b
            # Compute inverse quadratic interpolation
            m1 = (fa - fb) / (a - b)
            m2 = (fa - fc) / (a - c)
            num = fa * (m2 - m1)
            den = (fc - fb) * m1 * m2
            if den != 0:
                s = a - num / den
            else:
                s = a + 0.5 * (b - a)
        else:
            # Bisection
            s = a + 0.5 * (b - a)
        
        # Ensure s is in [a, b]
        if (s <= a + 0.5 * (b - a) and s >= b) or (s >= b - 0.5 * (b - a) and s <= a):
            s = a + 0.5 * (b - a)
        
        fs = f(s)
        if not math.isfinite(fs):
            # Fall back to bisection
            s = a + 0.5 * (b - a)
            fs = f(s)
            if not math.isfinite(fs):
                return None
        
        if fs == 0:
            return s
        
        if fa * fs < 0:
            b = s
            fb = fs
        else:
            c = a
            fc = fa
            a = s
            fa = fs
        
        d = b - a
        e = d
    
    return a if abs(fa) < abs(fb) else b


def _newton_polish(f, x, max_iter=20):
    """
    Newton refinement starting from x.
    """
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) <= 1e-12:
            return x
        h = 1e-8 * max(1.0, abs(x))
        d = (f(x + h) - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        if abs(x_new - x) < 1e-15 * max(1.0, abs(x)):
            break
        x = x_new
    return x
# EVOLVE-BLOCK-END
