"""Root finding for ill-conditioned problems using global search + safeguarded Newton."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
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

    def secant_method(x0, tol=TOL, max_iter=80):
        """Secant method with stall detection."""
        x1 = float(x0)
        h = 1e-5 * max(1.0, abs(x1))
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

        prev_fabs = abs(f2)
        stall_count = 0

        for i in range(max_iter):
            if not check_time():
                return False
            if abs(f2) <= tol:
                return True
            if abs(f1) <= tol:
                return True

            denom = f2 - f1
            if abs(denom) < 1e-30:
                h2 = 1e-5 * max(1.0, abs(x2)) * (1 + 0.1 * i)
                x2_new = x2 + h2
                try:
                    f2 = call_f(x2_new)
                except Exception:
                    return False
                x2 = x2_new
                denom = f2 - f1
                if abs(denom) < 1e-30:
                    return False

            x_new = x2 - f2 * (x2 - x1) / denom
            if not math.isfinite(x_new):
                return False
            if abs(x_new) > 1e15:
                return False

            curr_fabs = abs(f2)
            if curr_fabs > prev_fabs * 0.5:
                stall_count += 1
            else:
                stall_count = 0
            prev_fabs = curr_fabs

            if stall_count > 10:
                return False

            x1, f1 = x2, f2
            x2 = x_new
            try:
                f2 = call_f(x2)
            except Exception:
                return False

        return abs(f1) <= tol or abs(f2) <= tol

    def safeguarded_newton(x0, tol=TOL, max_iter=60):
        """
        Safeguarded Newton-Raphson with trust region.
        Uses central difference for derivative when possible.
        """
        x = float(x0)

        for i in range(max_iter):
            if not check_time():
                return False
            if eval_count >= max_evals - 5:
                return False

            try:
                fx = call_f(x)
            except Exception:
                return False
            if abs(fx) <= tol:
                return True

            # Central difference for derivative (more accurate for ill-conditioned)
            h = 1e-7 * max(1.0, abs(x))
            try:
                fp = call_f(x + h)
                fm = call_f(x - h)
            except Exception:
                return False

            d = (fp - fm) / (2 * h)
            if abs(d) < 1e-30:
                # Try forward difference with larger step
                h2 = 1e-4 * max(1.0, abs(x))
                try:
                    fp2 = call_f(x + h2)
                    d = (fp2 - fx) / h2
                except Exception:
                    return False
                if abs(d) < 1e-30:
                    return False

            step = fx / d

            # Trust region: limit step size
            max_step = 5.0 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = math.copysign(max_step, step)

            x_new = x - step
            if not math.isfinite(x_new):
                return False
            if abs(x_new) > 1e15:
                return False

            # Check if step improved things; if not, halve the step
            try:
                fx_new = call_f(x_new)
            except Exception:
                return False

            if abs(fx_new) <= tol:
                return True

            if abs(fx_new) > abs(fx) * 1.5:
                # Step made things worse; bisect between x and x_new
                x_mid = (x + x_new) / 2
                try:
                    fx_mid = call_f(x_mid)
                except Exception:
                    return False
                if abs(fx_mid) <= tol:
                    return True
                if abs(fx_mid) < abs(fx):
                    x = x_mid
                else:
                    x = x_new
            else:
                x = x_new

        return abs(fx) <= tol

    def bisection(x0, tol=TOL, max_iter=80):
        """Bisection with bracket search."""
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

        # Search right
        for i in range(30):
            x_test = x + step * (2 ** i)
            try:
                f_test = call_f(x_test)
            except Exception:
                break
            if not math.isfinite(f_test):
                break
            if fa is None:
                a, fa = x, fx
            if fa * f_test < 0:
                b, fb = x_test, f_test
                break

        # Search left if not found
        if b is None:
            a, b = None, None
            fa, fb = None, None
            for i in range(30):
                x_test = x - step * (2 ** i)
                try:
                    f_test = call_f(x_test)
                except Exception:
                    break
                if not math.isfinite(f_test):
                    break
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
            if abs(b - a) < 1e-15 * max(1.0, abs(mid)):
                break

        return min(abs(fa), abs(fb)) <= tol

    def global_search(x0, tol=TOL, max_points=80):
        """
        Global search: sample points in a geometrically expanding region around x0
        to find the point with minimum |f(x)|. This helps locate the basin of
        attraction for ill-conditioned problems where local methods may diverge.
        """
        x = float(x0)
        scale = max(1.0, abs(x))

        # Generate candidate points in expanding rings
        candidates = [x]

        # Ring 1: close to x0
        for h in [1e-4, 1e-3, 1e-2, 0.1]:
            candidates.append(x + h * scale)
            candidates.append(x - h * scale)

        # Ring 2: moderate distance
        for h in [0.5, 1.0, 2.0, 5.0]:
            candidates.append(x + h * scale)
            candidates.append(x - h * scale)

        # Ring 3: far
        for h in [10.0, 50.0, 100.0]:
            candidates.append(x + h * scale)
            candidates.append(x - h * scale)

        # Evaluate all candidates, track best
        local_best_x = x
        local_best_fabs = abs(fx) if 'fx' in dir() else best_fabs

        for xc in candidates:
            if eval_count >= max_evals - 10:
                break
            if not check_time():
                break
            if not math.isfinite(xc) or abs(xc) > 1e15:
                continue
            try:
                fc = call_f(xc)
            except Exception:
                continue
            if not math.isfinite(fc):
                continue
            fabs = abs(fc)
            if fabs < local_best_fabs:
                local_best_fabs = fabs
                local_best_x = xc
                if fabs <= tol:
                    return True

        # If we found a better point, try to polish it
        if local_best_fabs < best_fabs * 0.5:
            # Update global best
            best_x = local_best_x
            best_fabs = local_best_fabs
            if local_best_fabs <= tol:
                return True

        return local_best_fabs <= tol

    def polish_pass(x_start, tol=TOL, max_iter=30):
        """High-precision polish using secant method."""
        x1 = float(x_start)
        h = 1e-8 * max(1.0, abs(x1))
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
                h2 = 1e-8 * max(1.0, abs(x2)) * (1 + 0.01 * i)
                x2_new = x2 + h2
                try:
                    f2 = call_f(x2_new)
                except Exception:
                    return False
                x2 = x2_new
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

        return abs(f1) <= tol or abs(f2) <= tol

    def high_precision_polish(x_start, tol=TOL, max_iter=50):
        """Ultra-high-precision polish with multiple tiny step sizes."""
        x1 = float(x_start)
        step_sizes = [1e-10, 1e-11, 1e-12, 5e-11, 2e-10]

        best_x_local = x1
        best_fabs_local = abs(call_f(x1))

        if best_fabs_local <= tol:
            return True

        for h in step_sizes:
            if eval_count >= max_evals - 10:
                break
            if not check_time():
                break

            h_scaled = h * max(1.0, abs(x1))
            x2 = x1 + h_scaled

            try:
                f1 = call_f(x1)
                f2 = call_f(x2)
            except Exception:
                continue

            fabs1 = abs(f1)
            fabs2 = abs(f2)

            if fabs1 <= tol:
                return True
            if fabs2 <= tol:
                return True

            for i in range(10):
                if not check_time():
                    return False
                if eval_count >= max_evals - 5:
                    return False

                if abs(f2) <= tol:
                    return True
                if abs(f1) <= tol:
                    return True

                denom = f2 - f1
                if abs(denom) < 1e-35:
                    break

                x_new = x2 - f2 * (x2 - x1) / denom
                if not math.isfinite(x_new):
                    break
                if abs(x_new) > 1e15:
                    break

                x1, f1 = x2, f2
                x2 = x_new
                try:
                    f2 = call_f(x2)
                except Exception:
                    break

                fabs2 = abs(f2)
                if fabs2 < best_fabs_local:
                    best_fabs_local = fabs2
                    best_x_local = x2

            if best_fabs_local <= tol:
                return True

        return best_fabs_local <= tol

    # Main strategy: try multiple approaches in order of efficiency

    # Strategy 1: Quick secant from x0
    if secant_method(x0):
        return best_x

    # Strategy 2: Safeguarded Newton from x0
    if safeguarded_newton(x0):
        return best_x

    # Strategy 3: Global search to find a better starting point
    if eval_count < max_evals - 100:
        global_search(x0)

    # Strategy 4: Secant from the best point found by global search
    if best_fabs < TOL * 1000 and eval_count < max_evals - 50:
        if secant_method(best_x):
            return best_x

    # Strategy 5: Safeguarded Newton from the best point
    if best_fabs < TOL * 1000 and eval_count < max_evals - 50:
        if safeguarded_newton(best_x):
            return best_x

    # Strategy 6: Bisection
    if eval_count < max_evals - 80:
        if bisection(x0):
            return best_x

    # Strategy 7: Polish on best point
    if best_fabs < TOL * 1000 and best_fabs > 0 and eval_count < max_evals - 10:
        polish_pass(best_x)

    # Strategy 8: High-precision polish if very close
    if best_fabs < TOL * 10 and best_fabs > TOL and eval_count < max_evals - 15:
        high_precision_polish(best_x)

    return best_x
# EVOLVE-BLOCK-END
