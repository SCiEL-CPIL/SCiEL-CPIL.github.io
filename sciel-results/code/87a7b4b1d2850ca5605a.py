"""Root finding for ill-conditioned functions."""
import time
import math

try:
    from scipy.optimize import brentq
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Hybrid strategy within the bracketing + Illinois family:
      1. Adaptive step-doubling to secure a sign-change bracket (both directions).
      2. Illinois method (modified secant with fixed 0.5 shrink) for robust convergence.
      3. scipy brentq polish on the final bracket (exact local solve).
      4. Tighter secant polish pass to drive |f| well below tolerance.
    Includes a wall-clock deadline guard and a hard 500-evaluation cap.
    """
    x0 = float(x0)
    start_time = time.time()
    DEADLINE = 145.0
    MAX_EVALS = 500
    tol = 1e-8

    best_x = x0
    best_fval = float('inf')
    eval_count = 0

    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE

    def update_best(x, fx):
        nonlocal best_x, best_fval
        if abs(fx) < best_fval:
            best_fval = abs(fx)
            best_x = x
        if abs(fx) <= tol:
            return x
        return None

    def feval(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS or not elapsed_ok():
            return None
        eval_count += 1
        try:
            val = f(x)
        except Exception:
            return None
        if val is None:
            return None
        try:
            val = float(val)
        except Exception:
            return None
        if val != val:  # NaN
            return None
        if val == float('inf') or val == float('-inf'):
            return None
        return val

    # --- Initial evaluation ---
    fx0 = feval(x0)
    if fx0 is None:
        return best_x
    res = update_best(x0, fx0)
    if res is not None:
        return res

    # --- Bracket search: adaptive step-doubling in both directions ---
    a, fa = x0, fx0
    bracketed = False
    b, fb = None, None

    base_step = 1e-4 * max(1.0, abs(x0))
    step = base_step
    # Try a small set of initial offsets to catch near roots quickly
    for _ in range(60):
        if not elapsed_ok() or eval_count >= MAX_EVALS - 20:
            break

        # Positive direction
        bp = x0 + step
        fbp = feval(bp)
        if fbp is None:
            pass
        else:
            res = update_best(bp, fbp)
            if res is not None:
                return res
            if fa * fbp < 0:
                a, fa = x0, fx0
                b, fb = bp, fbp
                bracketed = True
                break

        # Negative direction
        bm = x0 - step
        fbm = feval(bm)
        if fbm is None:
            pass
        else:
            res = update_best(bm, fbm)
            if res is not None:
                return res
            if fa * fbm < 0:
                a, fa = x0, fx0
                b, fb = bm, fbm
                bracketed = True
                break

        step *= 2.0
        # Avoid runaway step
        if step > 1e15 * max(1.0, abs(x0)):
            break

    # --- Illinois method on bracket [a, b] ---
    if bracketed:
        # Ensure |fa| <= |fb|
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        for _ in range(300):
            if not elapsed_ok() or eval_count >= MAX_EVALS:
                break
            if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
                break
            if fb != fa:
                x_new = a - fa * (b - a) / (fb - fa)
            else:
                x_new = 0.5 * (a + b)
            fx_new = feval(x_new)
            if fx_new is None:
                break
            res = update_best(x_new, fx_new)
            if res is not None:
                return res
            # Update bracket
            if fa * fx_new < 0:
                b, fb = x_new, fx_new
            else:
                a, fa = x_new, fx_new
                # Illinois: shrink the replaced side by 0.5
                b = a + 0.5 * (b - a)
                fb = feval(b)
                if fb is None:
                    break
                res = update_best(b, fb)
                if res is not None:
                    return res
        # Final bracket endpoints
        res = update_best(a, fa)
        if res is not None:
            return res
        res = update_best(b, fb)
        if res is not None:
            return res

        # --- scipy brentq polish on the final bracket ---
        if HAVE_SCIPY and elapsed_ok() and eval_count < MAX_EVALS - 5:
            try:
                # brentq needs sign change; ensure it
                if fa * fb > 0:
                    # No sign change (shouldn't happen), skip
                    pass
                else:
                    x_root = brentq(f, a, b, xtol=1e-14, rtol=1e-14, maxiter=50)
                    fx_root = feval(x_root)
                    if fx_root is not None:
                        res = update_best(x_root, fx_root)
                        if res is not None:
                            return res
            except Exception:
                pass

        # --- Tighter secant polish ---
        # Use the two best points near the root
        x_prev, f_prev = a, fa
        x_curr, f_curr = b, fb
        for _ in range(30):
            if not elapsed_ok() or eval_count >= MAX_EVALS:
                break
            if abs(f_curr) <= tol:
                return x_curr
            if f_curr != f_prev:
                x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            else:
                break
            f_new = feval(x_new)
            if f_new is None:
                break
            res = update_best(x_new, f_new)
            if res is not None:
                return res
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_new, f_new

        return best_x
    else:
        # --- No bracket found: secant / Newton-like iteration ---
        x_prev = x0
        f_prev = fx0
        x_curr = x0 + 1e-6 * max(1.0, abs(x0))
        f_curr = feval(x_curr)
        if f_curr is None:
            return best_x
        res = update_best(x_curr, f_curr)
        if res is not None:
            return res

        for _ in range(300):
            if not elapsed_ok() or eval_count >= MAX_EVALS:
                break
            if f_curr == f_prev:
                # Degenerate: finite-difference Newton
                h = 1e-8 * max(1.0, abs(x_curr))
                f_plus = feval(x_curr + h)
                if f_plus is None:
                    break
                d = (f_plus - f_curr) / h
                if d == 0:
                    break
                x_new = x_curr - f_curr / d
            else:
                x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            # Guard against absurdly large steps
            if abs(x_new) > 1e15 * max(1.0, abs(x0)):
                break
            f_new = feval(x_new)
            if f_new is None:
                break
            res = update_best(x_new, f_new)
            if res is not None:
                return res
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_new, f_new

        return best_x

# EVOLVE-BLOCK-END
