import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    deadline = time.time() + 140.0  # safety margin before 150s limit
    call_count = [0]
    
    def fc(x):
        call_count[0] += 1
        if call_count[0] >= 495:  # hard cap
            raise RuntimeError("call limit")
        return f(x)
    
    x = float(x0)
    
    # First, check if x0 is already a root
    fx = fc(x)
    if abs(fx) <= 1e-8:
        return x
    
    # Try secant method for fast convergence (1 call per iteration after setup)
    x1 = x + 1e-4 * max(1.0, abs(x))
    fx1 = fc(x1)
    
    for i in range(50):
        if time.time() > deadline:
            break
        if abs(fx1) <= 1e-8:
            return x1
        if abs(fx) <= 1e-8:
            return x
        if fx == fx1:
            # Degenerate case, try a different step
            x2 = x + 1e-3 * max(1.0, abs(x))
            fx2 = fc(x2)
            if abs(fx2) <= 1e-8:
                return x2
            x, fx = x1, fx1
            x1, fx1 = x2, fx2
            continue
        x2 = x1 - fx1 * (x1 - x) / (fx1 - fx)
        # Keep x2 reasonable
        if abs(x2) > 1e10:
            x2 = x + 0.5 * (x1 - x)
        fx2 = fc(x2)
        x, fx = x1, fx1
        x1, fx1 = x2, fx2
    
    # If secant didn't converge, try bracketing + Brent's method
    # Search for a bracket around x0
    a, b = None, None
    fa, fb = None, None
    
    # Expand outward to find sign change
    step = max(1e-3, 0.1 * max(1.0, abs(x0)))
    for i in range(30):
        if time.time() > deadline:
            break
        if call_count[0] >= 480:
            break
        # Check left
        xa = x0 - step * (2 ** i)
        fa = fc(xa)
        if abs(fa) <= 1e-8:
            return xa
        if fx is not None and fa * fx < 0:
            a, b = xa, x
            fa, fb = fa, fx
            break
        # Check right
        xb = x0 + step * (2 ** i)
        fb = fc(xb)
        if abs(fb) <= 1e-8:
            return xb
        if fx is not None and fb * fx < 0:
            a, b = x, xb
            fa, fb = fx, fb
            break
        if fa is not None and fb is not None and fa * fb < 0:
            a, b = xa, xb
            fa, fb = fa, fb
            break
        if abs(step) > 1e8:
            break
    
    if a is None:
        # No bracket found, return best guess
        # Return the point with smallest |f(x)|
        if abs(fx) <= abs(fx1):
            return x
        return x1
    
    # Proper Brent's method
    a, b = float(a), float(b)
    fa, fb = float(fa), float(fb)
    if fa * fb > 0:
        # Shouldn't happen, but safety
        return a if abs(fa) < abs(fb) else b
    
    # Ensure |f(a)| <= |f(b)|
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    
    c = a
    fc_val = fa
    e = b - a
    
    for i in range(50):
        if time.time() > deadline:
            break
        if call_count[0] >= 495:
            break
        if abs(fa) <= 1e-8:
            return a
        if (b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
            return a
        
        # Inverse quadratic interpolation
        if a != b and b != c:
            # IQU formula from Numerical Recipes
            t1 = (a / (b - c)) * (fa / (fa - fc_val))
            t2 = (a / (a - c)) * (fa / (fa - fb))
            t3 = (b / (a - b)) * (fb / (fb - fa))
            s = t1 + t2 + t3
        elif a != b:
            # Linear interpolation
            s = a - fa * (b - a) / (fb - fa)
        else:
            s = a + 0.5 * (b - a)
        
        # Check if s is acceptable
        if ((s < min(a, b) or s > max(a, b)) or
            (abs(s - b) >= 0.5 * abs(b - c)) or
            (abs(s - a) < 1e-15)):
            s = 0.5 * (a + b)
        
        if abs(s - b) < 1e-15:
            s = b + 1e-15
        elif abs(s - a) < 1e-15:
            s = a + 1e-15
        
        fs = fc(s)
        c = b
        fc_val = fb
        
        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs
        
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        e = b - a
    
    return a
# EVOLVE-BLOCK-END
