"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START

def _counted_call(f, x, call_count):
    """Call f(x) and increment call_count (a list of one int). Returns (value, count)."""
    val = f(x)
    call_count[0] += 1
    return val, call_count[0]

def find_root(f, x0):
    """
    Hybrid root finder: Newton with adaptive FD derivative, bisection fallback,
    and deadline guard. Returns x with |f(x)| <= 1e-8 or best effort.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    call_count = [0]
    TOL = 1e-8
    TOL_STRICT = 1e-12
    MAX_CALLS = 450  # well under 500

    def elapsed():
        return time.time() - start_time

    def time_ok():
        return time.time() < deadline and call_count[0] < MAX_CALLS

    # --- Phase 1: Try Newton with adaptive step, track bracket for bisection ---
    x = float(x0)
    best_x = x
    best_fabs = float('inf')

    # Initial evaluation
    fx, _ = _counted_call(f, x, call_count)
    best_fabs = abs(fx)
    if best_fabs <= TOL_STRICT:
        return x

    # Track bracket: we need a sign change
    a, fa = x, fx
    b, fb = None, None

    # Try to find a bracket by expanding from x0
    # First, try a small perturbation to find opposite sign
    h0 = max(1e-4, 1e-4 * abs(x))
    for h_try in [h0, 2*h0, 4*h0, 8*h0, 16*h0, 32*h0, 64*h0]:
        if not time_ok():
            break
        xp = x + h_try
        fxp, _ = _counted_call(f, xp, call_count)
        if fa * fxp < 0:
            a, fa = x, fx
            b, fb = xp, fxp
            break
        xm = x - h_try
        fxm, _ = _counted_call(f, xm, call_count)
        if fa * fxm < 0:
            a, fa = x, fx
            b, fb = xm, fxm
            break
        # Update best
        if abs(fxp) < best_fabs:
            best_fabs = abs(fxp)
            best_x = xp
        if abs(fxm) < best_fabs:
            best_fabs = abs(fxm)
            best_x = xm

    # --- Phase 2: Newton's method with bisection safeguard ---
    # We'll use a combination: if we have a bracket, do safeguarded Newton (bisection fallback)
    # Otherwise, do pure Newton with damping

    x = float(x0)
    fx, _ = _counted_call(f, x, call_count)
    best_fabs = abs(fx)
    best_x = x

    if best_fabs <= TOL_STRICT:
        return x

    # Re-establish bracket if we lost it (we already have a, b from above potentially)
    # Actually let's restructure: use the bracket we found, or create one

    # Let's restart with a cleaner approach:
    # 1. Evaluate at x0
    # 2. Try to find bracket by expanding
    # 3. If bracket found: safeguarded Newton (Newton step, if it leaves bracket, bisect)
    # 4. If no bracket: damped Newton / secant

    # Reset
    x = float(x0)
    call_count[0] = 0  # reset for clean counting? No, the evaluator counts all calls.
    # Actually call_count is just for our internal budget. The evaluator counts separately.
    # But we need to be careful: the evaluator counts every f() call.
    # Let me re-read: "more than 500 calls" - so we must keep total calls <= 500.

    # Let me redo this more carefully.

    # --- Clean implementation ---
    x = float(x0)
    fx = f(x)
    call_count[0] = 1
    best_x = x
    best_fabs = abs(fx)

    if best_fabs <= TOL:
        return x

    # Find bracket
    a, b = None, None
    fa, fb = None, None

    # Try expanding around x0
    step = max(1e-3, 1e-3 * abs(x))
    for i in range(12):
        if call_count[0] >= MAX_CALLS or time.time() >= deadline:
            break
        xp = x + step
        fxp = f(xp)
        call_count[0] += 1
        if abs(fxp) < best_fabs:
            best_fabs = abs(fxp)
            best_x = xp
        if fx * fxp < 0:
            a, fa = x, fx
            b, fb = xp, fxp
            break

        xm = x - step
        fxm = f(xm)
        call_count[0] += 1
        if abs(fxm) < best_fabs:
            best_fabs = abs(fxm)
            best_x = xm
        if fx * fxm < 0:
            a, fa = x, fx
            b, fb = xm, fxm
            break

        step *= 2.0

    # If we have a bracket, use safeguarded Newton/bisection
    if a is not None and b is not None:
        # Safeguarded Newton: try Newton step, if it goes outside [a,b], bisect
        for _ in range(100):
            if call_count[0] >= MAX_CALLS or time.time() >= deadline:
                break
            if abs(b - a) < 1e-14 * max(1.0, abs(a)):
                break
            mid = 0.5 * (a + b)
            fmid = f(mid)
            call_count[0] += 1
            if abs(fmid) < best_fabs:
                best_fabs = abs(fmid)
                best_x = mid
            if abs(fmid) <= TOL:
                return mid

            # Newton step from midpoint
            h = 1e-6 * max(1.0, abs(mid))
            fmid_plus = f(mid + h)
            call_count[0] += 1
            d = (fmid_plus - fmid) / h
            if d != 0:
                xn = mid - fmid / d
            else:
                xn = mid

            # Check if Newton step is within bracket
            if a < xn < b:
                # Accept Newton step
                if fa * fmid < 0:
                    b, fb = mid, fmid
                else:
                    a, fa = mid, fmid
                # Evaluate at new point
                fxn = f(xn)
                call_count[0] += 1
                if abs(fxn) < best_fabs:
                    best_fabs = abs(fxn)
                    best_x = xn
                if abs(fxn) <= TOL:
                    return xn
                if fa * fxn < 0:
                    b, fb = xn, fxn
                else:
                    a, fa = xn, fxn
            else:
                # Bisect
                if fa * fmid < 0:
                    b, fb = mid, fmid
                else:
                    a, fa = mid, fmid

        # Return midpoint of final bracket or best
        if a is not None and b is not None:
            return 0.5 * (a + b)
        return best_x

    # No bracket found: use damped Newton / secant method
    # Use secant method as it doesn't need derivative
    x1 = x + max(1e-4, 1e-4 * abs(x))
    fx1 = f(x1)
    call_count[0] += 1
    if abs(fx1) < best_fabs:
        best_fabs = abs(fx1)
        best_x = x1
    if abs(fx1) <= TOL:
        return x1

    for _ in range(80):
        if call_count[0] >= MAX_CALLS or time.time() >= deadline:
            break
        if fx == fx1:
            break
        x2 = x1 - fx1 * (x1 - x) / (fx1 - fx)
        fx2 = f(x2)
        call_count[0] += 1
        if abs(fx2) < best_fabs:
            best_fabs = abs(fx2)
            best_x = x2
        if abs(fx2) <= TOL:
            return x2
        # Damping if step is too large
        if abs(x2 - x1) > 10 * max(1.0, abs(x1)):
            x2 = x1 + 0.5 * (x2 - x1)
            fx2 = f(x2)
            call_count[0] += 1
            if abs(fx2) < best_fabs:
                best_fabs = abs(fx2)
                best_x = x2
            if abs(fx2) <= TOL:
                return x2
        x, fx = x1, fx1
        x1, fx1 = x2, fx2

    return best_x

# EVOLVE-BLOCK-END
