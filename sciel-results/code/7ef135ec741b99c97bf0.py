"""Root finder for 3 fixed smooth polynomial equations with call budget optimization."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 for a smooth polynomial f.
    Uses a multi-strategy approach:
    1. Try Newton's method with finite-difference derivative (fast, ~2 calls/step)
    2. If Newton stalls, use secant method (2 calls/step)
    3. If still not converged, use bisection with bracket expansion
    4. Deadline guard to prevent timeout
    """
    start_time = time.time()
    DEADLINE = 10.0  # seconds, well under 120s limit with safety margin
    TOL = 1e-10  # tighter than required 1e-8
    
    def elapsed():
        return time.time() - start_time
    
    x = float(x0)
    best_x = x
    best_fabs = abs(f(x))
    
    # Phase 1: Newton's method with adaptive step
    # Use 2-point finite difference for derivative
    for i in range(40):
        if elapsed() > DEADLINE:
            break
        fx = f(x)
        fabs = abs(fx)
        if fabs < best_fabs:
            best_fabs = fabs
            best_x = x
        if fabs <= TOL:
            return x
        
        # Finite difference derivative (central if possible, forward otherwise)
        h = 1e-8 * max(1.0, abs(x))
        # Try central difference
        fp = f(x + h)
        fm = f(x - h)
        d = (fp - fm) / (2.0 * h)
        
        if d == 0 or not math.isfinite(d):
            break
        
        x_new = x - fx / d
        
        # Damping: if step is too large, damp it
        if not math.isfinite(x_new):
            break
        step = abs(x_new - x)
        if step > 10.0 * max(1.0, abs(x)):
            x_new = x + math.copysign(10.0 * max(1.0, abs(x)), x_new - x)
        
        x = x_new
    
    if elapsed() > DEADLINE:
        return best_x
    
    # Phase 2: Secant method (uses previous two points)
    # We need two points; use x0 and current x
    x0_sec = float(x0)
    x1_sec = x
    f0 = f(x0_sec)
    f1 = f(x1_sec)
    
    if abs(f1) < abs(f0):
        best_fabs = min(best_fabs, abs(f1))
        best_x = x1_sec
    if abs(f0) < abs(f1):
        best_fabs = min(best_fabs, abs(f0))
        best_x = x0_sec
    
    if abs(f1) <= TOL:
        return x1_sec
    if abs(f0) <= TOL:
        return x0_sec
    
    for i in range(40):
        if elapsed() > DEADLINE:
            break
        if f0 == f1:
            break
        x_new = x1_sec - f1 * (x1_sec - x0_sec) / (f1 - f0)
        if not math.isfinite(x_new):
            break
        f_new = f(x_new)
        if abs(f_new) < best_fabs:
            best_fabs = abs(f_new)
            best_x = x_new
        if abs(f_new) <= TOL:
            return x_new
        x0_sec = x1_sec
        f0 = f1
        x1_sec = x_new
        f1 = f_new
    
    if elapsed() > DEADLINE:
        return best_x
    
    # Phase 3: Bisection with bracket expansion
    # Try to find a bracket [a, b] where f(a) and f(b) have opposite signs
    a = x0
    b = x0
    fa = f(a)
    fb = f(b)
    
    # Expand bracket
    step_size = max(1.0, abs(x0)) * 0.5
    found_bracket = False
    for expand in range(20):
        if elapsed() > DEADLINE:
            break
        a = x0 - step_size
        b = x0 + step_size
        fa = f(a)
        fb = f(b)
        if fa * fb < 0:
            found_bracket = True
            break
        step_size *= 2.0
        if step_size > 1e6:
            break
    
    if found_bracket:
        for i in range(60):
            if elapsed() > DEADLINE:
                break
            mid = (a + b) / 2.0
            fm = f(mid)
            if abs(fm) < best_fabs:
                best_fabs = abs(fm)
                best_x = mid
            if abs(fm) <= TOL:
                return mid
            if fa * fm < 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
            if abs(b - a) < 1e-15 * max(1.0, abs(mid)):
                break
    
    return best_x
# EVOLVE-BLOCK-END
