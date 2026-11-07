"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
import time as _time

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using a call-efficient IQI/secant hybrid
    with smart bracketing that reuses prior evaluations.
    """
    start_time = _time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    TOL = 1e-8

    def _elapsed():
        return _time.time() - start_time

    def _check_deadline():
        return _elapsed() > DEADLINE

    # Cache of evaluations to avoid redundant calls
    cache = {}

    def _feval(x):
        if x in cache:
            return cache[x]
        val = f(x)
        cache[x] = val
        return val

    x0 = float(x0)
    fx0 = _feval(x0)
    if not math.isfinite(fx0):
        return x0
    if abs(fx0) <= TOL:
        return x0

    # Track best solution
    best_x = x0
    best_abs = abs(fx0)

    def _update_best(x, fx):
        nonlocal best_x, best_abs
        if math.isfinite(fx) and abs(fx) < best_abs:
            best_x = x
            best_abs = abs(fx)

    _update_best(x0, fx0)

    # --- Phase 1: Local IQI/secant hybrid ---
    # Build initial triple of points close to x0
    scale = max(1.0, abs(x0))
    h = 0.05 * scale

    # Choose 3 points: x0, x0+h, x0-h
    x1 = x0
    f1 = fx0
    x2 = x0 + h
    f2 = _feval(x2)
    if abs(f2) <= TOL:
        return x2
    _update_best(x2, f2)

    x3 = x0 - h
    f3 = _feval(x3)
    if abs(f3) <= TOL:
        return x3
    _update_best(x3, f3)

    # If we have a sign change in [x3, x2], great for bracketing later
    # Now run IQI/secant iterations
    # Points: a, b, c with f(a), f(b), f(c)
    # We want the point with smallest |f| to be the "target"
    def _iqi_step(a, fa, b, fb, c, fc):
        """Inverse quadratic interpolation. Returns new x estimate."""
        # Standard IQI formula
        # p = b - fb * [ (a-b)*fc*(fb-fc) / ((fa-fb)*(fa-fc))
        #               + (b-c)*fb*(fa-fc) / ((fb-fc)*(fa-fc)) ]
        fa_fb = fa - fb
        fa_fc = fa - fc
        fb_fc = fb - fc
        if fa_fb == 0 or fa_fc == 0 or fb_fc == 0:
            return None
        term1 = (a - b) * fc * (fb - fc) / (fa_fb * fa_fc)
        term2 = (b - c) * fb * (fa - fc) / (fb_fc * fa_fc)
        p = b - fb * (term1 + term2)
        return p

    # Order points so b has smallest |f|
    pts = [(x1, f1), (x2, f2), (x3, f3)]
    pts.sort(key=lambda t: abs(t[1]))
    a, fa = pts[1]
    b, fb = pts[0]
    c, fc = pts[2]

    for _ in range(30):
        if _check_deadline():
            break
        # IQI step
        p = _iqi_step(a, fa, b, fb, c, fc)
        if p is None or not math.isfinite(p):
            # Fallback to secant between b and a
            denom = fb - fa
            if denom == 0:
                break
            p = b - fb * (b - a) / denom
            if not math.isfinite(p):
                break

        # Keep p within reasonable bounds (don't go too far)
        span = max(abs(a - b), abs(b - c), 1e-12)
        if abs(p - b) > 10 * span:
            # Clamp to midpoint of the two closest points
            p = 0.5 * (b + a)

        fp = _feval(p)
        if not math.isfinite(fp):
            break
        if abs(fp) <= TOL:
            return p
        _update_best(p, fp)

        # Update triple: drop the point with largest |f| among (a, c, p)
        # Keep b and the two best of {a, c, p}
        candidates = [(a, fa), (c, fc), (p, fp)]
        candidates.sort(key=lambda t: abs(t[1]))
        a, fa = candidates[0]
        c, fc = candidates[1]
        # b stays as the best
        # But we need b to be distinct from a and c; if collision, adjust
        if a == b or c == b:
            # Reorder
            all_pts = [(a, fa), (b, fb), (c, fc)]
            all_pts.sort(key=lambda t: abs(t[1]))
            a, fa = all_pts[1]
            b, fb = all_pts[0]
            c, fc = all_pts[2]

    _update_best(b, fb)

    # --- Phase 2: Bracketing (reuses cache) ---
    # Try to find a sign-change bracket around x0 using cached + new evals
    fa0 = fx0
    bracket_a = None
    bracket_b = None

    # Check cached points for sign changes with x0
    for xp, fxp in cache.items():
        if xp == x0:
            continue
        if not math.isfinite(fxp):
            continue
        if fa0 * fxp < 0:
            bracket_a = x0
            bracket_b = xp
            break

    if bracket_a is None:
        # Expand outward with geometric steps
        h = 0.1 * scale
        max_h = 1e6 * scale
        found = False
        while h <= max_h:
            if _check_deadline():
                break
            for sign in (1.0, -1.0):
                xp = x0 + sign * h
                fxp = _feval(xp)
                if abs(fxp) <= TOL:
                    return xp
                _update_best(xp, fxp)
                if fa0 * fxp < 0:
                    bracket_a = x0
                    bracket_b = xp
                    found = True
                    break
            if found:
                break
            h *= 2.0

    if bracket_a is None:
        # No bracket found; return best
        return best_x

    # --- Phase 3: Brent's method on the bracket ---
    def _brentq(f_func, a_val, b_val, tol=1e-14, maxiter=100):
        fa_val = f_func(a_val)
        fb_val = f_func(b_val)
        if fa_val == 0.0:
            return a_val
        if fb_val == 0.0:
            return b_val
        if fa_val * fb_val > 0:
            raise ValueError("no sign change")

        c = a_val
        fc = fa_val
        d = b_val - a_val
        e = d

        for _ in range(maxiter):
            if _check_deadline():
                break
            if fb_val == 0.0:
                return b_val
            if fa_val == 0.0:
                return a_val
            if fa_val * fb_val > 0:
                a_val = b_val
                fa_val = fb_val
                d = e = b_val - a_val
            elif abs(fc) < abs(fb_val):
                a_val = b_val
                fa_val = fb_val
                b_val = c
                fb_val = fc
                c = d
                fc = fa_val
                d = e = b_val - a_val

            tol1 = 2.220446049250313e-16 * max(1.0, abs(b_val))
            m = 0.5 * (a_val - b_val)
            if abs(m) <= tol1 or fb_val == 0.0:
                return b_val

            if abs(e) >= tol1 and abs(fa_val) > abs(fb_val):
                s = fb_val / fa_val
                if a_val == c:
                    p = 2.0 * m * s
                    q = 1.0 - s
                else:
                    q = fa_val / fc
                    r = fb_val / fc
                    p = s * (2.0 * m * q * (q - r) - (b_val - c) * (r - 1.0))
                    q = (q - 1.0) * (r - 1.0) * (s - 1.0)
                if p > 0.0:
                    q = -q
                elif p < 0.0:
                    p = -p
                if 2.0 * p < min(3.0 * m * q - abs(tol1 * q), abs(e * q)):
                    e = d
                    d = p / q
                else:
                    d = m
                    e = d
            else:
                d = m
                e = d

            if abs(d) < tol1:
                d = tol1 if d > 0 else -tol1

            c = b_val
            fc = fb_val
            b_val = b_val + d
            fb_val = f_func(b_val)

        return b_val

    try:
        root = _brentq(_feval, bracket_a, bracket_b)
        if math.isfinite(root):
            froot = _feval(root)
            if abs(froot) <= TOL:
                return root
            _update_best(root, froot)
            # Polish with a few secant steps if not tight enough
            if abs(froot) > TOL:
                # Use two nearby points for secant polish
                # Pick the two closest cached points to root
                nearby = sorted(cache.items(), key=lambda kv: abs(kv[0] - root))
                if len(nearby) >= 2:
                    (xa, faa) = nearby[0]
                    (xb, fbb) = nearby[1]
                    if faa != fbb and math.isfinite(faa) and math.isfinite(fbb):
                        for _ in range(8):
                            if _check_deadline():
                                break
                            denom = fbb - faa
                            if denom == 0:
                                break
                            xn = xb - fbb * (xb - xa) / denom
                            if not math.isfinite(xn):
                                break
                            fn = _feval(xn)
                            if abs(fn) <= TOL:
                                return xn
                            _update_best(xn, fn)
                            xa, faa = xb, fbb
                            xb, fbb = xn, fn
            return best_x if abs(froot) > TOL else root
    except Exception:
        pass

    # --- Phase 4: Final high-precision polish using Newton-like steps ---
    # If we have a good candidate, try to refine it further
    if best_abs > TOL:
        # Try a few Newton steps using numerical derivative
        x_new = best_x
        f_new = best_abs
        dx = 1e-6 * max(1.0, abs(x_new))
        x_plus = x_new + dx
        f_plus = _feval(x_plus)
        if math.isfinite(f_plus):
            dfdx = (f_plus - f_new) / dx
            if abs(dfdx) > 1e-15:
                for _ in range(5):
                    if _check_deadline():
                        break
                    x_candidate = x_new - f_new / dfdx
                    if not math.isfinite(x_candidate):
                        break
                    f_candidate = _feval(x_candidate)
                    if not math.isfinite(f_candidate):
                        break
                    if abs(f_candidate) <= TOL:
                        return x_candidate
                    _update_best(x_candidate, f_candidate)
                    # Update derivative estimate
                    dx = 1e-6 * max(1.0, abs(x_candidate))
                    x_plus = x_candidate + dx
                    f_plus = _feval(x_plus)
                    if math.isfinite(f_plus):
                        dfdx = (f_plus - f_candidate) / dx
                        x_new = x_candidate
                        f_new = f_candidate
                    else:
                        break
                    if abs(dfdx) < 1e-15:
                        break

    return best_x
# EVOLVE-BLOCK-END
