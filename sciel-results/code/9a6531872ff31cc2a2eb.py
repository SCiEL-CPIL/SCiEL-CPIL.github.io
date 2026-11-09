import math
import time
import numpy as np

def find_root(f, x0):
    """
    Find root x of f(x) such that |f(x)| <= 1e-8.
    Structural change: DFT-based spectral root-finding.
    1. Sample f at equally spaced points around x0 to estimate dominant frequency/phase.
    2. Predict nearest zero-crossing from fitted sinusoid.
    3. Newton/secant refinement from predicted point.
    4. Bracketing fallback with IQI/bisection hybrid.
    """
    start_time = time.time()
    deadline = start_time + 10.0
    tol_final = 1e-8
    max_iter = 100

    x0 = float(x0)
    f0 = f(x0)
    if abs(f0) <= tol_final:
        return x0
    if not math.isfinite(f0):
        base_step = 1e-4 * max(1.0, abs(x0))
        for s in [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64]:
            x_try = x0 + s * base_step
            f_try = f(x_try)
            if math.isfinite(f_try) and abs(f_try) <= tol_final:
                return x_try
            if math.isfinite(f_try):
                x0, f0 = x_try, f_try
                break
        if not math.isfinite(f0):
            return x0

    best_x = x0
    best_f = abs(f0)

    # Phase 1: DFT-based spectral root-finding
    # Sample at multiple window sizes to find the dominant oscillation period
    # Try several window widths; for each, sample N points and do DFT
    spectral_guesses = []
    
    # Window sizes to try (multiples of a base scale)
    base_scale = 1.0  # We'll adapt based on results
    window_multipliers = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    
    for wm in window_multipliers:
        if time.time() > deadline:
            break
        # Sample N points over a window of width 2*wm*base_scale centered at x0
        N = 16  # Number of samples
        window_width = wm * base_scale
        if window_width < 1e-6:
            window_width = 1e-6
        xs = np.linspace(x0 - window_width, x0 + window_width, N)
        try:
            ys = np.array([f(float(xi)) for xi in xs])
        except Exception:
            continue
        if not np.all(np.isfinite(ys)):
            continue
        
        # Remove DC component
        y_centered = ys - np.mean(ys)
        
        # DFT
        Y = np.fft.fft(y_centered)
        # Only look at positive frequencies (first half)
        freqs = np.fft.fftfreq(N, d=2.0*window_width/N)
        # Get the dominant frequency (excluding DC)
        # Use magnitude, look at indices 1 to N//2
        magnitudes = np.abs(Y[1:N//2+1])
        if len(magnitudes) == 0:
            continue
        dom_idx = np.argmax(magnitudes) + 1
        dom_freq = freqs[dom_idx]
        dom_mag = magnitudes[dom_idx - 1]
        
        # Check if this is a significant oscillation
        total_mag = np.sum(magnitudes)
        if total_mag < 1e-12 or dom_mag < 0.1 * total_mag:
            continue
        
        # The dominant frequency tells us the period
        if abs(dom_freq) < 1e-12:
            continue
        period = 1.0 / abs(dom_freq)
        
        # Phase: arg(Y[dom_idx])
        phase = np.angle(Y[dom_idx])
        
        # The signal is approximately: A*cos(2*pi*dom_freq*x + phase)
        # Zeros occur when 2*pi*dom_freq*x + phase = pi/2 + k*pi
        # x = (pi/2 + k*pi - phase) / (2*pi*dom_freq)
        
        # Find the zero closest to x0
        # 2*pi*dom_freq*x0 + phase = theta0
        theta0 = 2.0 * math.pi * dom_freq * x0 + phase
        # We want theta = pi/2 + k*pi closest to theta0
        # k = round((theta0 - pi/2) / pi)
        k = round((theta0 - math.pi/2) / math.pi)
        x_zero = (math.pi/2 + k * math.pi - phase) / (2.0 * math.pi * dom_freq)
        
        if math.isfinite(x_zero):
            # Evaluate f at the predicted zero
            f_zero = f(float(x_zero))
            if math.isfinite(f_zero):
                spectral_guesses.append((x_zero, f_zero, abs(f_zero)))
                if abs(f_zero) < best_f:
                    best_x, best_f = x_zero, abs(f_zero)
                if abs(f_zero) <= tol_final:
                    return x_zero
        
        # Also try the adjacent zero (k+1 and k-1)
        for dk in [1, -1]:
            k2 = k + dk
            x_zero2 = (math.pi/2 + k2 * math.pi - phase) / (2.0 * math.pi * dom_freq)
            if math.isfinite(x_zero2):
                f_zero2 = f(float(x_zero2))
                if math.isfinite(f_zero2):
                    if abs(f_zero2) < best_f:
                        best_x, best_f = x_zero2, abs(f_zero2)
                    if abs(f_zero2) <= tol_final:
                        return x_zero2
                    spectral_guesses.append((x_zero2, f_zero2, abs(f_zero2)))
    
    # Phase 2: Newton/secant refinement from best spectral guess
    if spectral_guesses:
        # Sort by |f| and try Newton from the best
        spectral_guesses.sort(key=lambda t: t[2])
        x_start = spectral_guesses[0][0]
        f_start = spectral_guesses[0][1]
        
        # Secant iteration from the best guess
        # Use a small perturbation to get a second point
        h = 1e-6 * max(1.0, abs(x_start))
        x_prev = x_start - h
        f_prev = f(x_prev)
        x_curr = x_start
        f_curr = f_start
        
        if not math.isfinite(f_prev):
            x_prev = x_start + h
            f_prev = f(x_prev)
            x_curr = x_start
            f_curr = f_start
        
        if math.isfinite(f_prev):
            for _ in range(30):
                if time.time() > deadline:
                    break
                denom = f_curr - f_prev
                if abs(denom) < 1e-15:
                    break
                x_next = x_curr - f_curr * (x_curr - x_prev) / denom
                if not math.isfinite(x_next):
                    break
                f_next = f(x_next)
                if not math.isfinite(f_next):
                    break
                if abs(f_next) < best_f:
                    best_x, best_f = x_next, abs(f_next)
                if abs(f_next) <= tol_final:
                    return x_next
                x_prev, f_prev = x_curr, f_curr
                x_curr, f_curr = x_next, f_next
        
        if abs(f_curr) <= tol_final:
            return x_curr
        if abs(f_prev) <= tol_final:
            return x_prev

    # Phase 3: Bracketing with hybrid method (bisection + inverse quadratic)
    # Try to find a sign change around x0
    a, fa = x0, f0
    base_step2 = 1e-4 * max(1.0, abs(x0))
    bracket_found = False
    b, fb = None, None
    
    for s in [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64, 128, -128, 256, -256, 512, -512, 1024, -1024, 2048, -2048]:
        if time.time() > deadline:
            break
        x_try = x0 + s * base_step2
        f_try = f(x_try)
        if not math.isfinite(f_try):
            continue
        if abs(f_try) <= tol_final:
            return x_try
        if abs(f_try) < best_f:
            best_x, best_f = x_try, abs(f_try)
        if fa * f_try < 0:
            a, fa = x0, f0
            b, fb = x_try, f_try
            bracket_found = True
            break
    
    if not bracket_found:
        # Try using spectral guesses to find a bracket
        for (xg, fg, _) in spectral_guesses[:5]:
            if time.time() > deadline:
                break
            for s in [1, -1, 2, -2, 4, -4, 8, -8]:
                h = s * 1e-4 * max(1.0, abs(xg))
                x_try = xg + h
                f_try = f(x_try)
                if not math.isfinite(f_try):
                    continue
                if abs(f_try) <= tol_final:
                    return x_try
                if fg * f_try < 0:
                    a, fa = xg, fg
                    b, fb = x_try, f_try
                    bracket_found = True
                    break
            if bracket_found:
                break
    
    if not bracket_found:
        return best_x

    # Hybrid: Inverse Quadratic Interpolation with Bisection fallback
    m = (a + b) / 2.0
    fm = f(m)
    if not math.isfinite(fm):
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if abs(fa) <= tol_final:
                return a
            if abs(fb) <= tol_final:
                return b
            if abs(b - a) < 1e-15:
                return (a + b) / 2.0
            mid = (a + b) / 2.0
            fmid = f(mid)
            if not math.isfinite(fmid):
                break
            if abs(fmid) <= tol_final:
                return mid
            if fa * fmid < 0:
                b, fb = mid, fmid
            else:
                a, fa = mid, fmid
        return a if abs(fa) < abs(fb) else b

    if abs(fm) <= tol_final:
        return m

    x1, f1 = a, fa
    x2, f2 = m, fm
    x3, f3 = b, fb

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if abs(f1) <= tol_final:
            return x1
        if abs(f2) <= tol_final:
            return x2
        if abs(f3) <= tol_final:
            return x3
        if abs(x3 - x1) < 1e-15:
            return (x1 + x3) / 2.0

        if f1 * f2 < 0:
            x_lo, f_lo, x_hi, f_hi = x1, f1, x2, f2
            x_mid, f_mid = x3, f3
        elif f2 * f3 < 0:
            x_lo, f_lo, x_hi, f_hi = x2, f2, x3, f3
            x_mid, f_mid = x1, f1
        else:
            mid = (x1 + x3) / 2.0
            fmid = f(mid)
            if not math.isfinite(fmid):
                break
            if abs(fmid) <= tol_final:
                return mid
            if f1 * fmid < 0:
                x3, f3 = mid, fmid
            else:
                x1, f1 = mid, fmid
            continue

        denom1 = (f_lo - f_mid) * (f_lo - f_hi)
        denom2 = (f_mid - f_lo) * (f_mid - f_hi)
        denom3 = (f_hi - f_lo) * (f_hi - f_mid)

        x_iqi = None
        if abs(denom1) > 1e-30 and abs(denom2) > 1e-30 and abs(denom3) > 1e-30:
            x_iqi = (x_lo * f_mid * f_hi / denom1 +
                     x_mid * f_lo * f_hi / denom2 +
                     x_hi * f_lo * f_mid / denom3)

        if (x_iqi is not None and math.isfinite(x_iqi)
                and x_lo < x_iqi < x_hi):
            x_new = x_iqi
        else:
            x_new = (x_lo + x_hi) / 2.0

        f_new = f(x_new)
        if not math.isfinite(f_new):
            x_new = (x_lo + x_hi) / 2.0
            f_new = f(x_new)
            if not math.isfinite(f_new):
                break

        if abs(f_new) <= tol_final:
            return x_new

        if f_lo * f_new < 0:
            x_hi, f_hi = x_new, f_new
        else:
            x_lo, f_lo = x_new, f_new

        x1, f1 = x_lo, f_lo
        x3, f3 = x_hi, f_hi
        x2, f2 = x_mid, f_mid

    # Final polish
    if time.time() < deadline:
        best_a, best_b = None, None
        best_fa, best_fb = None, None
        pts = [(x1, f1), (x2, f2), (x3, f3)]
        for i in range(3):
            for j in range(i+1, 3):
                if pts[i][1] * pts[j][1] < 0:
                    if best_a is None or abs(pts[i][1]) + abs(pts[j][1]) < abs(best_fa) + abs(best_fb):
                        best_a, best_fa = pts[i]
                        best_b, best_fb = pts[j]
        if best_a is not None and best_b is not None:
            a2, fa2 = best_a
            b2, fb2 = best_b
            for _ in range(20):
                if time.time() > deadline:
                    break
                if abs(fa2) <= tol_final:
                    return a2
                if abs(fb2) <= tol_final:
                    return b2
                if abs(b2 - a2) < 1e-15:
                    return (a2 + b2) / 2.0
                mid2 = (a2 + b2) / 2.0
                fm2 = f(mid2)
                if not math.isfinite(fm2):
                    break
                if abs(fm2) <= tol_final:
                    return mid2
                if fa2 * fm2 < 0:
                    b2, fb2 = mid2, fm2
                else:
                    a2, fa2 = mid2, fm2
            return a2 if abs(fa2) < abs(fb2) else b2

    candidates = [(x1, f1), (x2, f2), (x3, f3)]
    best = min(candidates, key=lambda t: abs(t[1]))
    return best[0]
