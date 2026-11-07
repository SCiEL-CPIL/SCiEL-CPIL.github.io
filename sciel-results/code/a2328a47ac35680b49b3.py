import math
import time
from scipy.optimize import brentq

_DEADLINE = time.time() + 140.0
_CALL_BUDGET = 480

def find_root(f, x0):
    call_count = [0]
    best_x = float(x0)
    best_fabs = float('inf')

    def safe_call(x):
        if call_count[0] >= _CALL_BUDGET:
            raise RuntimeError("budget")
        if time.time() > _DEADLINE:
            raise RuntimeError("deadline")
        call_count[0] += 1
        try:
            val = f(x)
        except Exception:
            val = float('nan')
        if math.isfinite(val):
            fabs = abs(val)
            if fabs < best_fabs:
                best_fabs = fabs
                best_x = x
            return val
        return float('nan')

    x = float(x0)
    # Try Newton's method first (2 calls per step)
    newton_success = False
    for _ in range(15):
        try:
            fx = safe_call(x)
            if not math.isfinite(fx):
                break
            if abs(fx) <= 1e-12:
                newton_success = True
                break
            h = 1e-7 * max(1.0, abs(x))
            fx2 = safe_call(x + h)
            if not math.isfinite(fx2):
                break
            d = (fx2 - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            # Damping if step is too large
            if abs(x_new - x) > 10 * max(1.0, abs(x)):
                x_new = x + 10 * max(1.0, abs(x)) * (1 if x_new > x else -1)
            x = x_new
        except RuntimeError:
            break

    if newton_success:
        return x

    # Try to find a bracket and use Brent's method
    # Scan outward from x0 to find a sign change
    a = None
    b = None
    fa = None
    fb = None
    step = 1.0
    for i in range(50):
        if call_count[0] >= _CALL_BUDGET - 20:
            break
        for sign in (1, -1):
            x_try = x0 + sign * step
            try:
                fx_try = safe_call(x_try)
                if not math.isfinite(fx_try):
                    continue
                if a is None:
                    a = x_try
                    fa = fx_try
                else:
                    b = x_try
                    fb = fx_try
                    if fa * fb < 0:
                        break
            except RuntimeError:
                break
        else:
            step *= 2
            continue
        break

    if a is not None and b is not None and fa * fb < 0:
        try:
            # Use brentq with maxiter to limit calls
            # brentq typically uses ~10-30 calls for well-behaved functions
            remaining = _CALL_BUDGET - call_count[0]
            if remaining > 5:
                result = brentq(safe_call, a, b, maxiter=max(10, remaining // 3), xtol=1e-14, rtol=1e-14)
                if math.isfinite(result):
                    return result
        except Exception:
            pass

    # Fallback: return best x found
    if math.isfinite(best_x):
        return best_x
    return float(x0)
