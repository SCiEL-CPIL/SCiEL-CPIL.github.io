"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0
    max_evals = 500
    TOL = 1e-8
    eval_count = 0
    best_x = float(x0)
    best_fabs = abs(f(best_x))
    eval_count += 1

    def call_f(x):
        nonlocal eval_count, best_x, best_fabs
        if eval_count >= max_evals:
            raise Exception("max evals")
        if time.time() > deadline:
            raise Exception("deadline")
        val = f(x)
        eval_count += 1
        if not (isinstance(val, (int, float)) and math.isfinite(val)):
            val = float('nan')
        fabs = abs(val)
        if math.isfinite(fabs) and fabs < best_fabs:
            best_fabs = fabs
            best_x = x
        return val

    def check_time():
        return time.time() < deadline

    def try_secant(x0, tol=TOL, max_iter=150):
        x1 = float(x0)
        h = 1e-3 * max(1.0, abs(x1))
        x2 = x1 + h
        try:
            f1 = call_f(x1)
            f2 = call_f(x2)
        except Exception:
            return False
        if abs(f1) <= tol:
            return True
        if abs(f2) <= tol:
            return True
        for i in range(max_iter):
            if not check_time():
                return False
            if abs(f2) <= tol:
                return True
            if abs(f1) <= tol:
                return True
            denom = f2 - f1
            if abs(denom) < 1e-30:
                return False
            x_new = x2 - f2 * (x2 - x1) / denom
            if not math.isfinite(x_new):
                return False
            if abs(x_new) > 1e15:
                return False
            x1, f1 = x2, f2
            x2 = x_new
            try:
                f2 = call_f(x2)
            except Exception:
                return False
        return abs(f2) <= tol or abs(f1) <= tol

    def try_halley(x0, tol=TOL, max_iter=80):
        """Halley's method using forward differences for f' and f''."""
        x = float(x0)
        h1 = 1e-5 * max(1.0, abs(x))
        for i in range(max_iter):
            if not check_time():
                return False
            try:
                fx = call_f(x)
            except Exception:
                return False
            if abs(fx) <= tol:
                return True
            try:
                fp = call_f(x + h1)
                fm = call_f(x - h1)
            except Exception:
                return False
            d1 = (fp - fm) / (2 * h1)
            if abs(d1) < 1e-30:
                return False
            d2 = (fp - 2 * fx + fm) / (h1 * h1)
            # Halley's step: x_new = x - 2*f*d1 / (2*d1^2 - f*d2)
            denom = 2 * d1 * d1 - fx * d2
            if abs(denom) < 1e-30:
                return False
            step = 2 * fx * d1 / denom
            # Limit step size for stability
            if abs(step) > 5.0 * max(1.0, abs(x)):
                step = math.copysign(5.0 * max(1.0, abs(x)), step)
            x_new = x - step
            if not math.isfinite(x_new):
                return False
            if abs(x_new) > 1e15:
                return False
            x = x_new
        return False

    def try_bisection(x0, tol=TOL, max_iter=200):
        x = float(x0)
        try:
            fx = call_f(x)
        except Exception:
            return False
        if abs(fx) <= tol:
            return True

        step = 1e-3 * max(1.0, abs(x))
        a, b = None, None
        fa, fb = None, None

        for direction in [1, -1]:
            for i in range(35):
                try:
                    x_test = x + direction * step * (2 ** i)
                    f_test = call_f(x_test)
                except Exception:
                    break
                if not math.isfinite(f_test):
                    break
                if direction > 0:
                    if fa is None:
                        a, fa = x, fx
                    if fa * f_test < 0:
                        b, fb = x_test, f_test
                        break
                else:
                    if fb is None:
                        b, fb = x, fx
                    if fb * f_test < 0:
                        a, fa = x_test, f_test
                        break

        if a is None or b is None:
            return False

        for i in range(max_iter):
            if not check_time():
                return False
            mid = (a + b) / 2
            try:
                fm = call_f(mid)
            except Exception:
                return False
            if abs(fm) <= tol:
                return True
            if fa * fm <= 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            if abs(b - a) < 1e-14 * max(1.0, abs(mid)):
                break

        return min(abs(fa), abs(fb)) <= tol

    def try_newton_forward(x0, tol=TOL, max_iter=100):
        x = float(x0)
        for i in range(max_iter):
            if not check_time():
                return False
            try:
                fx = call_f(x)
            except Exception:
                return False
            if abs(fx) <= tol:
                return True
            h = 1e-7 * max(1.0, abs(x))
            try:
                fp = call_f(x + h)
            except Exception:
                return False
            d = (fp - fx) / h
            if abs(d) < 1e-30:
                return False
            step = fx / d
            if abs(step) > 10.0:
                step = math.copysign(10.0, step)
            x_new = x - step
            if not math.isfinite(x_new):
                return False
            if abs(x_new) > 1e15:
                return False
            x = x_new
        return False

    # Try secant first (most robust, 1 eval/iter after initial 2)
    if try_secant(x0):
        return best_x

    # Try Halley's method (3 evals/iter, cubic convergence)
    if try_halley(x0):
        return best_x

    # Try Newton with forward difference (2 evals/iter)
    if try_newton_forward(x0):
        return best_x

    # Try bisection with wider bracket search
    if try_bisection(x0):
        return best_x

    # Final polish: if we're close, use secant with tighter tolerance
    if best_fabs < 1e-4 and eval_count < max_evals - 20:
        # Try a few more secant steps from best_x
        x1 = best_x
        h = 1e-6 * max(1.0, abs(x1))
        x2 = x1 + h
        try:
            f1 = call_f(x1)
            f2 = call_f(x2)
        except Exception:
            pass
        else:
            if abs(f1) <= TOL:
                return best_x
            if abs(f2) <= TOL:
                return best_x
            for _ in range(50):
                if not check_time():
                    break
                if abs(f2) <= TOL:
                    return best_x
                if abs(f1) <= TOL:
                    return best_x
                denom = f2 - f1
                if abs(denom) < 1e-30:
                    break
                x_new = x2 - f2 * (x2 - x1) / denom
                if not math.isfinite(x_new) or abs(x_new) > 1e15:
                    break
                x1, f1 = x2, f2
                x2 = x_new
                try:
                    f2 = call_f(x2)
                except Exception:
                    break

    return best_x
# EVOLVE-BLOCK-END
