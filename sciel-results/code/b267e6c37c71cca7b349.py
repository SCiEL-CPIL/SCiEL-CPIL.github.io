"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def find_root(f, x0):
    import math

    evals = [0]

    def feval(x):
        evals[0] += 1
        return f(x)

    x0 = float(x0)
    fx0 = feval(x0)
    if abs(fx0) <= 1e-8:
        return x0

    # --- Attempt to find a sign-change bracket ---
    a = b = None
    fa = fb = None

    # Expand right
    step = 1e-3 * max(1.0, abs(x0))
    fa, a = fx0, x0
    for _ in range(30):
        if evals[0] >= 480:
            break
        xt = a + step
        ft = feval(xt)
        if abs(ft) <= 1e-8:
            return xt
        if fa * ft < 0:
            b, fb = xt, ft
            break
        a, fa = xt, ft
        step *= 2.0

    if b is None:
        # Expand left
        step = 1e-3 * max(1.0, abs(x0))
        fa, a = fx0, x0
        for _ in range(30):
            if evals[0] >= 480:
                break
            xt = a - step
            ft = feval(xt)
            if abs(ft) <= 1e-8:
                return xt
            if fa * ft < 0:
                b, fb = xt, ft
                break
            a, fa = xt, ft
            step *= 2.0

    if b is not None:
        # --- Safeguarded Newton within bracket [a,b] ---
        for _ in range(500 - evals[0]):
            if evals[0] >= 495:
                break
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
                break

            mid = 0.5 * (a + b)
            fmid = feval(mid)
            if abs(fmid) <= 1e-8:
                return mid

            # Update bracket with midpoint
            if fa * fmid < 0:
                b, fb = mid, fmid
            else:
                a, fa = mid, fmid

            # Newton step from the endpoint with smaller |f|
            if abs(fa) < abs(fb):
                xc, fc = a, fa
            else:
                xc, fc = b, fb

            h = 1e-8 * max(1.0, abs(xc))
            fp = feval(xc + h)
            fm = feval(xc - h)
            d = (fp - fm) / (2.0 * h)

            if abs(d) > 1e-30:
                xn = xc - fc / d
                if a < xn < b:
                    fn = feval(xn)
                    if abs(fn) <= 1e-8:
                        return xn
                    if fa * fn < 0:
                        b, fb = xn, fn
                    else:
                        a, fa = xn, fn

        # Return best of endpoints
        if abs(fa) < abs(fb):
            return a
        return b
    else:
        # --- No bracket: minimize |f| locally ---
        best_x, best_abs = x0, abs(fx0)
        base = max(1.0, abs(x0))
        for scale in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e4]:
            for sgn in (1, -1):
                if evals[0] >= 480:
                    break
                xt = x0 + sgn * scale * base
                ft = feval(xt)
                if abs(ft) < best_abs:
                    best_abs = abs(ft)
                    best_x = xt
                if abs(ft) <= 1e-8:
                    return xt

        # Newton with damping from best_x
        x = best_x
        fx = feval(x)
        for _ in range(500 - evals[0]):
            if evals[0] >= 495:
                break
            if abs(fx) <= 1e-8:
                return x
            h = 1e-6 * max(1.0, abs(x))
            d = (feval(x + h) - fx) / h
            if abs(d) < 1e-30:
                break
            step_size = -fx / d
            # Damp if step too large
            max_step = 5.0 * max(1.0, abs(x))
            if abs(step_size) > max_step:
                step_size = max_step * (1.0 if step_size > 0 else -1.0)
            xn = x + step_size
            fn = feval(xn)
            if abs(fn) < abs(fx):
                x, fx = xn, fn
            else:
                # Halve
                xn = x + 0.5 * step_size
                fn = feval(xn)
                if abs(fn) < abs(fx):
                    x, fx = xn, fn
                else:
                    break
        return x

# EVOLVE-BLOCK-END
