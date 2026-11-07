import numpy as np
from scipy.optimize import brentq, minimize_scalar
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    x = float(x0)
    try:
        fx = f(x)
        if abs(fx) <= 1e-8:
            return x
        if not np.isfinite(fx):
            return x
    except:
        return x

    # Search for a bracket
    h = 1e-5
    max_h = 1e5
    fa = fx
    a = x
    
    while h <= max_h:
        if time.time() > deadline:
            break
            
        # Check positive direction
        x_pos = x + h
        try:
            f_pos = f(x_pos)
            if abs(f_pos) <= 1e-8:
                return x_pos
            if np.isfinite(f_pos) and fa * f_pos < 0:
                # Found bracket [x, x_pos]
                try:
                    root = brentq(f, a, x_pos, xtol=1e-12, rtol=1e-12, maxiter=50)
                    if abs(f(root)) <= 1e-8:
                        return root
                except:
                    pass
        except:
            pass

        # Check negative direction
        x_neg = x - h
        try:
            f_neg = f(x_neg)
            if abs(f_neg) <= 1e-8:
                return x_neg
            if np.isfinite(f_neg) and fa * f_neg < 0:
                # Found bracket [x_neg, x]
                try:
                    root = brentq(f, x_neg, a, xtol=1e-12, rtol=1e-12, maxiter=50)
                    if abs(f(root)) <= 1e-8:
                        return root
                except:
                    pass
        except:
            pass

        h *= 10.0

    # Fallback: minimize |f(x)| in a local window
    # If we didn't find a sign change, maybe the root is a touch (tangent) or we missed it.
    # Try bounded minimization.
    bounds = (x - 1e2, x + 1e2)
    try:
        res = minimize_scalar(lambda t: abs(f(t)), bounds=bounds, method='bounded', options={'xatol': 1e-10})
        if res.success:
            x_min = res.x
            f_min = f(x_min)
            if abs(f_min) <= 1e-8:
                return x_min
            # Try to polish with Newton if derivative is available or approximated
            # But if it's close, maybe just return x_min if it's the best we have
            # However, if it's not <= 1e-8, we might fail. 
            # Let's try a few Newton steps from x_min
            x_curr = x_min
            for _ in range(10):
                fx_curr = f(x_curr)
                if abs(fx_curr) <= 1e-8:
                    return x_curr
                h_new = 1e-6 * max(1.0, abs(x_curr))
                d = (f(x_curr + h_new) - fx_curr) / h_new
                if d == 0:
                    break
                x_new = x_curr - fx_curr / d
                if not np.isfinite(x_new):
                    break
                x_curr = x_new
            if abs(f(x_curr)) <= 1e-8:
                return x_curr
            return x_min
    except:
        pass

    return x
