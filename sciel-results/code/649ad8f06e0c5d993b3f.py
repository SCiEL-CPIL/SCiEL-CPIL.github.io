"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def _fit_sin(y, dx):
    """Estimate amplitude, frequency (cycles/unit) and phase from samples y on grid step dx."""
    n = len(y)
    c = (n - 1) / 2.0
    # Remove DC offset
    ym = [y[k] - y[c] for k in range(n)]
    # DFT at each integer bin k=1..n-1, take the dominant non-DC bin
    best_k, best_mag = 0, 0.0
    for k in range(1, n):
        re = 0.0
        im = 0.0
        for j in range(n):
            ang = 2.0 * math.pi * k * j / n
            re += ym[j] * math.cos(ang)
            im -= ym[j] * math.sin(ang)
        mag = re * re + im * im
        if mag > best_mag:
            best_mag = mag
            best_k = k
    if best_k == 0 or best_mag <= 0.0:
        return 0.0, 0.0, 0.0
    amp = math.sqrt(best_mag) * 2.0 / n
    freq = best_k / (n * dx)          # cycles per unit
    # phase: y ~ A cos(2*pi*freq*x + phi); at sample j, arg_j = 2*pi*freq*(j - c)*dx + phi
    # recover phi from the dominant DFT coefficient
    re = 0.0
    im = 0.0
    for j in range(n):
        ang = 2.0 * math.pi * best_k * j / n
        re += ym[j] * math.cos(ang)
        im -= ym[j] * math.sin(ang)
    phase_at_j = math.atan2(-im, re)
    arg_c = 2.0 * math.pi * freq * (-c) * dx
    phi = phase_at_j - arg_c
    return amp, freq, phi

def find_root(f, x0):
    x = float(x0)
    if not math.isfinite(x):
        x = 0.0
    fx = f(x)
    if not math.isfinite(fx):
        fx = 0.0
    if abs(fx) <= 1e-8:
        return x

    # 1) Spectral stage: sample on a grid around x0, fit a sinusoid, predict a zero.
    try:
        dx = max(1e-4, 1e-3 * max(1.0, abs(x)))
        n = 25
        half = (n - 1) / 2.0
        xs = [x + (j - half) * dx for j in range(n)]
        ys = [f(v) for v in xs]
        amp, freq, phi = _fit_sin(ys, dx)
        if amp > 1e-30 and freq > 0.0:
            w = 2.0 * math.pi * freq
            # solve cos(w*t + phi) = 0  ->  w*t + phi = pi/2 + m*pi
            base = (math.pi / 2.0 - phi) / w
            period = 1.0 / freq
            # nearest zero to t=0
            m0 = round(-base / period)
            t = base + m0 * period
            cand = x + t
            if math.isfinite(cand):
                fc = f(cand)
                if math.isfinite(fc) and abs(fc) <= 1e-8:
                    return cand
                # Use cand as the new starting point for local refinement
                x = cand
                fx = fc
    except Exception:
        pass

    # 2) Local damped Newton refinement with bisection safeguard.
    best_x = x
    best_f = fx
    if math.isfinite(fx) and abs(fx) < abs(best_f):
        best_f = fx
        best_x = x

    a = x
    fa = fx
    # Establish a bracket by expanding from x if signs differ.
    b = x
    fb = None
    if math.isfinite(fa):
        step = max(1e-4, 1e-3 * max(1.0, abs(x)))
        for _ in range(60):
            b = x + step
            try:
                fb = f(b)
            except Exception:
                fb = None
            if fb is None or not math.isfinite(fb):
                b = x - step
                try:
                    fb = f(b)
                except Exception:
                    fb = None
                if fb is None or not math.isfinite(fb):
                    break
            if math.isfinite(fb):
                if abs(fb) < abs(best_f):
                    best_f = fb
                    best_x = b
                if abs(fb) <= 1e-8:
                    return b
                if fa * fb < 0:
                    break
            step *= 2.0
    else:
        # No finite value at x; just return best finite seen.
        return best_x

    # Damped Newton within bracket [a,b]
    for _ in range(60):
        if abs(fa) <= 1e-8:
            return a
        if fb is not None and abs(fb) <= 1e-8:
            return b
        # Newton step from a
        h = 1e-6 * max(1.0, abs(a))
        try:
            d = (f(a + h) - fa) / h
        except Exception:
            d = 0.0
        if d == 0 or not math.isfinite(d):
            # fall back to bisection
            if fb is not None and math.isfinite(fb) and fa * fb < 0:
                mid = 0.5 * (a + b)
                fm = f(mid)
                if not math.isfinite(fm):
                    break
                if abs(fm) <= 1e-8:
                    return mid
                if fm * fa < 0:
                    b = mid
                    fb = fm
                else:
                    a = mid
                    fa = fm
                continue
            break
        step = fa / d
        new = a - step
        if not math.isfinite(new):
            break
        # Damping: keep within bracket if one exists.
        if fb is not None and math.isfinite(fb) and fa * fb < 0:
            if not (min(a, b) <= new <= max(a, b)):
                new = 0.5 * (a + b)
        try:
            fnew = f(new)
        except Exception:
            fnew = None
        if fnew is None or not math.isfinite(fnew):
            break
        if abs(fnew) < abs(best_f):
            best_f = fnew
            best_x = new
        if abs(fnew) <= 1e-8:
            return new
        if fb is not None and math.isfinite(fb):
            if fa * fnew < 0:
                b = new
                fb = fnew
            else:
                a = new
                fa = fnew
        else:
            a = new
            fa = fnew
    if math.isfinite(best_x):
        return best_x
    return a
# EVOLVE-BLOCK-END
