import math
import time

def find_root(f, x0):
    """
    Find root x of f(x) such that |f(x)| <= 1e-8.
    Uses DFT-based spectral estimation to predict nearest zero-crossing,
    then Newton/secant refinement, with bracketing fallback.
    """
    start_time = time.time()
    deadline = start_time + 10.0
    tol = 1e-12
    max_iter = 100

    x0 = float(x0)
    f0 = f(x0)
    if abs(f0) <= tol:
        return x0
    if not math.isfinite(f0):
        base_step = 1e-4 * max(1.0, abs(x0))
        for s in [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64]:
            x_try = x0 + s * base_step
            f_try = f(x_try)
            if math.isfinite(f_try) and abs(f_try) <= tol:
                return x_try
            if math.isfinite(f_try):
                x0, f0 = x_try, f_try
                break
        if not math.isfinite(f0):
            return x0

    # Phase 0: DFT-based spectral estimation to predict nearest zero-crossing
    # Sample N points in a window around x0
    n_samples = 16
    # Window size: use a scale that captures at least one full period if possible
    # Start with a small window, expand if needed
    window_scale = 1e-3 * max(1.0, abs(x0))
    
    best_x = x0
    best_f = abs(f0)
    
    # Try several window sizes to capture the oscillation
    for wscale in [1e-3, 1e-2, 1e-1, 1.0]:
        if time.time() > deadline:
            break
        half_win = wscale * max(1.0, abs(x0))
        # Sample n_samples points evenly in [x0 - half_win, x0 + half_win]
        xs = []
        fs = []
        for i in range(n_samples):
            t = i / (n_samples - 1)
            x_sample = (x0 - half_win) + t * 2 * half_win
            f_sample = f(x_sample)
            if math.isfinite(f_sample):
                xs.append(x_sample)
                fs.append(f_sample)
            else:
                xs.append(x_sample)
                fs.append(0.0)  # Mark as invalid
        
        # Check if we have enough valid samples
        valid_count = sum(1 for fv in fs if math.isfinite(fv))
        if valid_count < 8:
            continue
        
        # Compute DFT to estimate dominant frequency
        # Use the first n_samples valid samples
        valid_x = []
        valid_f = []
        for xv, fv in zip(xs, fs):
            if math.isfinite(fv):
                valid_x.append(xv)
                valid_f.append(fv)
        
        n = len(valid_f)
        if n < 8:
            continue
        
        # Remove mean
        mean_f = sum(valid_f) / n
        centered_f = [fv - mean_f for fv in valid_f]
        
        # Compute DFT for a few frequency bins
        # The sampling interval
        dx = (valid_x[-1] - valid_x[0]) / (n - 1)
        
        # Try frequency bins corresponding to periods within the window
        # We want to find the dominant oscillation frequency
        best_power = 0
        best_k = 0
        # Check k from 1 to n//2
        for k in range(1, n // 2 + 1):
            # DFT coefficient
            re = 0.0
            im = 0.0
            for i in range(n):
                angle = 2 * math.pi * k * i / n
                re += centered_f[i] * math.cos(angle)
                im -= centered_f[i] * math.sin(angle)
            power = re * re + im * im
            if power > best_power:
                best_power = power
                best_k = k
        
        if best_k == 0 or best_power < 1e-30:
            continue
        
        # Dominant frequency: k cycles over the window
        # Period T = window_length / k
        window_length = valid_x[-1] - valid_x[0]
        if window_length <= 0:
            continue
        T = window_length / best_k
        
        # Phase: from the DFT coefficient
        re = 0.0
        im = 0.0
        for i in range(n):
            angle = 2 * math.pi * best_k * i / n
            re += centered_f[i] * math.cos(angle)
            im -= centered_f[i] * math.sin(angle)
        
        # The signal is approximately A*cos(2*pi*x/T + phi) + mean
        # At x = valid_x[0], the phase is 2*pi*0/T + phi = phi
        # But our DFT gives us the complex coefficient
        # x(t) = Re(C * exp(i*2*pi*k*t/n)) where C = re + i*im
        # Actually, let's use the amplitude and phase from the DFT
        amplitude = math.sqrt(re * re + im * im) / n
        if amplitude < 1e-15:
            continue
        
        # Phase at the first sample
        phase = math.atan2(-im, re)
        
        # The signal model: f(x) ≈ mean_f + amplitude * cos(2*pi*(x - valid_x[0])/T + phase)
        # We want to find x where f(x) = 0
        # mean_f + amplitude * cos(2*pi*(x - valid_x[0])/T + phase) = 0
        # cos(2*pi*(x - valid_x[0])/T + phase) = -mean_f / amplitude
        ratio = -mean_f / amplitude
        if abs(ratio) > 1.0:
            continue
        
        # Solve for the phase angle
        base_angle = math.acos(ratio)
        
        # Two solutions per period:
        # 2*pi*(x - valid_x[0])/T + phase = ±base_angle + 2*pi*m
        # x = valid_x[0] + T/(2*pi) * (±base_angle - phase + 2*pi*m)
        
        # Find the nearest zero-crossing to x0
        candidates = []
        for sign in [1, -1]:
            for m in range(-2, 3):
                x_candidate = valid_x[0] + T / (2 * math.pi) * (sign * base_angle - phase + 2 * math.pi * m)
                candidates.append(x_candidate)
        
        # Find the candidate closest to x0
        if candidates:
            nearest_x = min(candidates, key=lambda xc: abs(xc - x0))
            f_nearest = f(nearest_x)
            if math.isfinite(f_nearest):
                if abs(f_nearest) < best_f:
                    best_x, best_f = nearest_x, abs(f_nearest)
                if abs(f_nearest) <= tol:
                    return nearest_x
        
        # If we found a good candidate, break out of window size loop
        if best_f < abs(f0) * 0.5:
            break
    
    # Phase 1: Secant/Newton refinement from the best point found
    x_prev = best_x
    f_prev = f(x_prev)
    if abs(f_prev) <= tol:
        return x_prev
    
    # Use a small perturbation for secant
    base_step = 1e-8 * max(1.0, abs(x_prev))
    x_curr = x_prev + base_step
    f_curr = f(x_curr)
    if abs(f_curr) <= tol:
        return x_curr
    if not math.isfinite(f_curr):
        x_curr = x_prev - base_step
        f_curr = f(x_curr)
        if abs(f_curr) <= tol:
            return x_curr
        if not math.isfinite(f_curr):
            # Fall back to original x0
            x_prev, f_prev = x0, f0
            x_curr = x0 + base_step
            f_curr = f(x_curr)
            if not math.isfinite(f_curr):
                x_curr = x0 - base_step
                f_curr = f(x_curr)
    
    if not math.isfinite(f_curr):
        # Can't do secant; go to bracketing
        a, fa = x_prev, f_prev
        b, fb = x0, f0
    else:
        a, fa = x_prev, f_prev
        b, fb = x_curr, f_curr
    
    # Secant iterations
    for _ in range(30):
        if time.time() > deadline:
            break
        if not math.isfinite(fa) or not math.isfinite(fb):
            break
        denom = fb - fa
        if abs(denom) < 1e-15:
            break
        x_next = b - fb * (b - a) / denom
        if not math.isfinite(x_next):
            break
        f_next = f(x_next)
        if not math.isfinite(f_next):
            break
        if abs(f_next) < best_f:
            best_x, best_f = x_next, abs(f_next)
        if abs(f_next) <= tol:
            return x_next
        a, fa = b, fb
        b, fb = x_next, f_next
    
    if abs(fb) <= tol:
        return b
    if abs(fa) <= tol:
        return a
    
    # Phase 2: Bracketing with hybrid method
    if fa * fb > 0:
        # No sign change; expand to find one
        if abs(fa) < abs(fb):
            a, b = a, b
        else:
            a, b = b, a
        fa, fb = f(a), f(b)
        
        base_step2 = 1e-4 * max(1.0, abs(a))
        bracket_found = False
        for s in [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64, 128, -128, 256, -256, 512, -512]:
            if time.time() > deadline:
                break
            x_try = a + s * base_step2
            f_try = f(x_try)
            if not math.isfinite(f_try):
                continue
            if abs(f_try) <= tol:
                return x_try
            if fa * f_try < 0:
                if abs(fa) < abs(f_try):
                    a, b, fa, fb = a, x_try, fa, f_try
                else:
                    a, b, fa, fb = x_try, a, f_try, fa
                bracket_found = True
                break
        
        if not bracket_found:
            return best_x
    
    # Bisection/IQI hybrid
    m = (a + b) / 2.0
    fm = f(m)
    if not math.isfinite(fm):
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if abs(fa) <= tol:
                return a
            if abs(fb) <= tol:
                return b
            if abs(b - a) < 1e-15:
                return (a + b) / 2.0
            mid = (a + b) / 2.0
            fmid = f(mid)
            if not math.isfinite(fmid):
                break
            if abs(fmid) <= tol:
                return mid
            if fa * fmid < 0:
                b, fb = mid, fmid
            else:
                a, fa = mid, fmid
        return a if abs(fa) < abs(fb) else b
    
    if abs(fm) <= tol:
        return m
    
    # IQI with bisection fallback
    x1, f1 = a, fa
    x2, f2 = m, fm
    x3, f3 = b, fb
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if abs(f1) <= tol:
            return x1
        if abs(f2) <= tol:
            return x2
        if abs(f3) <= tol:
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
            if abs(fmid) <= tol:
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
        
        if abs(f_new) <= tol:
            return x_new
        
        if f_lo * f_new < 0:
            x_hi, f_hi = x_new, f_new
        else:
            x_lo, f_lo = x_new, f_new
        
        x_mid, f_mid = x_hi if x_mid == x_lo else x_lo, f_hi if x_mid == x_lo else f_mid
        x1, f1 = x_lo, f_lo
        x3, f3 = x_hi, f_hi
        x2, f2 = x_mid, f_mid
    
    candidates = [(x1, f1), (x2, f2), (x3, f3)]
    best = min(candidates, key=lambda t: abs(t[1]))
    return best[0]
