import math
import time
import numpy as np

def find_root(f, x0):
    """
    Find root x of f(x) such that |f(x)| <= 1e-8.
    Uses DFT-based spectral estimation to predict zero-crossings,
    then secant/Newton refinement for high precision.
    """
    start_time = time.time()
    deadline = start_time + 10.0
    tol_final = 1e-8

    x0 = float(x0)
    f0 = f(x0)
    if math.isfinite(f0) and abs(f0) <= tol_final:
        return x0

    # Handle non-finite f0
    if not math.isfinite(f0):
        base_step = 1e-4 * max(1.0, abs(x0))
        for s in [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64]:
            if time.time() > deadline:
                break
            x_try = x0 + s * base_step
            f_try = f(x_try)
            if math.isfinite(f_try) and abs(f_try) <= tol_final:
                return x_try
            if math.isfinite(f_try):
                x0, f0 = x_try, f_try
                break
        if not math.isfinite(f0):
            return x0

    # Step 1: DFT-based spectral estimation to find dominant frequency and phase
    # Sample at multiple scales to capture oscillatory behavior
    base_step = max(1e-3, 1e-2 * max(1.0, abs(x0)))
    
    # Try several sample windows to find a good frequency estimate
    best_candidate = None
    best_candidate_f = float('inf')
    
    # Sample at a few different window sizes
    for window_scale in [1.0, 0.1, 0.01, 10.0, 100.0]:
        if time.time() > deadline:
            break
        h = base_step * window_scale
        # Sample 16 points over a window of size ~20*h centered on x0
        n_samples = 16
        x_samples = np.linspace(x0 - 10*h, x0 + 10*h, n_samples)
        f_samples = []
        all_finite = True
        for x_s in x_samples:
            if time.time() > deadline:
                break
            f_s = f(float(x_s))
            if not math.isfinite(f_s):
                all_finite = False
                break
            f_samples.append(f_s)
            if abs(f_s) <= tol_final:
                return float(x_s)
        
        if not all_finite or len(f_samples) < 8:
            continue
        
        f_arr = np.array(f_samples)
        # Remove DC component
        f_arr = f_arr - np.mean(f_arr)
        
        # Compute FFT
        F = np.fft.rfft(f_arr)
        freqs = np.fft.rfftfreq(n_samples, d=h)
        
        # Find dominant frequency (skip DC)
        if len(F) > 1:
            magnitudes = np.abs(F[1:])
            if np.max(magnitudes) > 0:
                idx = np.argmax(magnitudes)
                dominant_freq = freqs[1 + idx]
                phase = np.angle(F[1 + idx])
                
                # Predict zero crossings from the fitted sinusoid
                # f(x) ≈ A * sin(2π * freq * (x - x0) + phase)
                # Zero when 2π * freq * (x - x0) + phase = kπ
                # x = x0 + (kπ - phase) / (2π * freq)
                
                # Find nearest few zero crossings
                for k in range(-5, 6):
                    x_zero = x0 + (k * math.pi - phase) / (2 * math.pi * dominant_freq)
                    if not math.isfinite(x_zero):
                        continue
                    f_zero = f(float(x_zero))
                    if not math.isfinite(f_zero):
                        continue
                    if abs(f_zero) <= tol_final:
                        return float(x_zero)
                    if abs(f_zero) < best_candidate_f:
                        best_candidate_f = abs(f_zero)
                        best_candidate = float(x_zero)
    
    # Step 2: If we have a good candidate from DFT, refine with secant
    if best_candidate is not None and time.time() < deadline:
        xc = best_candidate
        fc = f(xc)
        if math.isfinite(fc) and abs(fc) <= tol_final:
            return xc
        
        # Get a second point for secant
        h_refine = max(1e-6, 1e-4 * max(1.0, abs(xc)))
        x2 = xc + h_refine
        f2 = f(x2)
        
        if math.isfinite(f2):
            # Secant iterations
            xa, fa = xc, fc
            xb, fb = x2, f2
            for _ in range(50):
                if time.time() > deadline:
                    break
                denom = fb - fa
                if abs(denom) < 1e-15:
                    break
                xn = xb - fb * (xb - xa) / denom
                if not math.isfinite(xn):
                    break
                fn = f(xn)
                if not math.isfinite(fn):
                    break
                if abs(fn) <= tol_final:
                    return xn
                if abs(xn - xb) < 1e-15:
                    break
                xa, fa = xb, fb
                xb, fb = xn, fn
            
            if abs(fb) <= tol_final:
                return xb
            if abs(fa) <= tol_final:
                return xa
            return xb if abs(fb) < abs(fa) else xa
    
    # Step 3: Fallback to local model + secant (original approach)
    base_step = 1e-6 * max(1.0, abs(x0))
    xs = [x0]
    fs = [f0]
    for s in [1, -1, 2, -2]:
        if time.time() > deadline:
            break
        x_try = x0 + s * base_step
        f_try = f(x_try)
        if math.isfinite(f_try):
            xs.append(x_try)
            fs.append(f_try)
            if abs(f_try) <= tol_final:
                return x_try

    best_x = xs[0]
    best_f = abs(fs[0])
    for i in range(1, len(xs)):
        if abs(fs[i]) < best_f:
            best_x = xs[i]
            best_f = abs(fs[i])

    if len(xs) >= 2:
        idx_sorted = sorted(range(len(xs)), key=lambda i: abs(fs[i]))
        i1, i2 = idx_sorted[0], idx_sorted[1]
        xa, fa = xs[i1], fs[i1]
        xb, fb = xs[i2], fs[i2]
        
        for _ in range(30):
            if time.time() > deadline:
                break
            denom = fb - fa
            if abs(denom) < 1e-15:
                break
            xn = xb - fb * (xb - xa) / denom
            if not math.isfinite(xn):
                break
            fn = f(xn)
            if not math.isfinite(fn):
                break
            if abs(fn) < best_f:
                best_x = xn
                best_f = abs(fn)
            if abs(fn) <= tol_final:
                return xn
            if abs(xn - xb) < 1e-15:
                break
            xa, fa = xb, fb
            xb, fb = xn, fn

        if abs(fb) <= tol_final:
            return xb
        if abs(fa) <= tol_final:
            return xa
    
    return best_x
