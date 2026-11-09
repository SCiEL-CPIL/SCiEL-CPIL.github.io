import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find root x of f(x) such that |f(x)| <= 1e-8.
    
    Strategy:
    1. Evaluate f(x0). If good, return.
    2. Sample f at N equally spaced points around x0 to estimate dominant
       oscillation frequency and phase via DFT.
    3. Analytically predict nearest zero-crossing from fitted sinusoid.
    4. Switch to local Newton/secant refinement from predicted point.
    5. If spectral prediction fails, fall back to bracketing + hybrid method.
    """
    start_time = time.time()
    deadline = start_time + 8.0
    
    tol_final = 1e-8
    
    x0 = float(x0)
    
    # --- Phase 0: Initial Evaluation ---
    f0 = f(x0)
    if not math.isfinite(f0):
        # Try nearby points
        best_x = x0
        best_f = float('inf')
        base_step = 1e-4 * max(1.0, abs(x0))
        found_finite = False
        for s in [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64, 128, -128, 256, -256, 512, -512]:
            if time.time() > deadline:
                return best_x if math.isfinite(best_f) else x0
            x_try = x0 + s * base_step
            f_try = f(x_try)
            if math.isfinite(f_try):
                if not found_finite:
                    x0, f0 = x_try, f_try
                    found_finite = True
                    break
                if abs(f_try) < abs(best_f):
                    best_x, best_f = x_try, f_try
            if abs(f_try) <= tol_final:
                return x_try
        if not found_finite:
            return x0
            
    if abs(f0) <= tol_final:
        return x0

    best_x = x0
    best_f = abs(f0)

    # --- Phase 1: DFT-based Spectral Root Finding ---
    # Sample function at N equally spaced points to estimate dominant frequency
    N = 16  # Number of samples for DFT
    # Choose a reasonable window size based on x0 scale
    window = max(1.0, abs(x0)) * 0.5
    # Try multiple window sizes to find the dominant oscillation
    best_spectral_x = None
    best_spectral_f = float('inf')
    
    for window_scale in [0.5, 1.0, 2.0, 5.0, 10.0]:
        if time.time() > deadline:
            break
        w = max(1e-3, abs(x0) * window_scale * 0.1 + 1e-3)
        # Sample points: x0 + (i - N/2) * dx where dx = 2*w/N
        dx = 2.0 * w / N
        xs = x0 + (np.arange(N) - N // 2) * dx
        fs = np.array([f(float(xi)) for xi in xs])
        
        # Check if all finite
        if not np.all(np.isfinite(fs)):
            continue
        
        # Check if any sample is already a root
        min_idx = np.argmin(np.abs(fs))
        if abs(fs[min_idx]) <= tol_final:
            return float(xs[min_idx])
        if abs(fs[min_idx]) < best_spectral_f:
            best_spectral_f = abs(fs[min_idx])
            best_spectral_x = float(xs[min_idx])
        
        # Compute DFT to find dominant frequency
        F = np.fft.fft(fs)
        # Get magnitudes (ignore DC component at index 0)
        magnitudes = np.abs(F[1:N//2])  # Positive frequencies
        if len(magnitudes) == 0:
            continue
        # Find dominant frequency (excluding DC)
        k_dom = np.argmax(magnitudes) + 1  # +1 to account for skipping index 0
        if magnitudes[k_dom - 1] < 1e-10:
            continue  # No significant oscillation
        
        # Dominant frequency: k_dom cycles over N samples
        # Period in sample index: N / k_dom
        # Period in x: dx * N / k_dom
        period_x = dx * N / k_dom
        
        # Get phase from the complex DFT coefficient
        phase = np.angle(F[k_dom])
        
        # The function is approximately: A * cos(2*pi*k_dom*x/N + phase)
        # where A = |F[k_dom]| / (N/2) for real signals
        # Zero crossings occur when cos(theta) = 0, i.e., theta = pi/2 + n*pi
        # 2*pi*k_dom*(x - x0)/N + phase = pi/2 + n*pi
        # (x - x0) = N/(2*pi*k_dom) * (pi/2 + n*pi - phase)
        
        # Find nearest zero crossing to x0
        # theta(x) = 2*pi*k_dom*(x - x0)/N + phase
        # We want theta = pi/2 + n*pi for integer n
        # x = x0 + N/(2*pi*k_dom) * (pi/2 + n*pi - phase)
        
        # Find n that minimizes |x - x0|
        # n = (2*pi*k_dom*(x0 - x0)/N + phase - pi/2) / pi (approximately)
        # For x near x0, theta(x0) = phase
        # Nearest zero: n such that pi/2 + n*pi is closest to phase (mod 2*pi)
        
        # Normalize phase to [0, 2*pi)
        phase_norm = phase % (2 * math.pi)
        
        # Zeros at theta = pi/2, 3*pi/2, 5*pi/2, ...
        # i.e., theta = (2m+1)*pi/2 for integer m
        # Find m that minimizes |(2m+1)*pi/2 - phase_norm|
        
        best_m = 0
        best_dist = float('inf')
        for m in range(-10, 11):
            theta_target = (2 * m + 1) * math.pi / 2
            # Account for wrapping
            diff = (theta_target - phase_norm) % (2 * math.pi)
            diff = min(diff, 2 * math.pi - diff)
            if diff < best_dist:
                best_dist = diff
                best_m = m
        
        # Predicted zero crossing
        theta_zero = (2 * best_m + 1) * math.pi / 2
        x_pred = x0 + N / (2 * math.pi * k_dom) * (theta_zero - phase)
        
        if not math.isfinite(x_pred):
            continue
        
        f_pred = f(float(x_pred))
        if not math.isfinite(f_pred):
            continue
        
        if abs(f_pred) < best_spectral_f:
            best_spectral_f = abs(f_pred)
            best_spectral_x = float(x_pred)
        
        if abs(f_pred) <= tol_final:
            return float(x_pred)
        
        # Also try the next zero crossing in each direction
        for dm in [-1, 1]:
            m_try = best_m + dm
            theta_try = (2 * m_try + 1) * math.pi / 2
            x_try = x0 + N / (2 * math.pi * k_dom) * (theta_try - phase)
            if math.isfinite(x_try):
                f_try = f(float(x_try))
                if math.isfinite(f_try):
                    if abs(f_try) < best_spectral_f:
                        best_spectral_f = abs(f_try)
                        best_spectral_x = float(x_try)
                    if abs(f_try) <= tol_final:
                        return float(x_try)
    
    # If we found a good spectral prediction, refine it
    if best_spectral_x is not None and abs(best_spectral_f) < abs(f0):
        x0 = best_spectral_x
        f0 = f(x0)
        if abs(f0) <= tol_final:
            return x0
        best_x = x0
        best_f = abs(f0)
    
    # --- Phase 2: Secant Method for Local Refinement ---
    base_step = 1e-5 * max(1.0, abs(x0))
    
    x_prev = x0
    f_prev = f0
    x_curr = x0 + base_step
    f_curr = f(x_curr)
    
    if not math.isfinite(f_curr):
        x_curr = x0 - base_step
        f_curr = f(x_curr)
        if not math.isfinite(f_curr):
            x_curr = x0 + base_step * 10
            f_curr = f(x_curr)
            if not math.isfinite(f_curr):
                return best_x

    if abs(f_curr) <= tol_final:
        return x_curr
    if abs(f_curr) < best_f:
        best_x = x_curr
        best_f = abs(f_curr)

    max_secant_iter = 20
    for _ in range(max_secant_iter):
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
            best_x = x_next
            best_f = abs(f_next)
            
        if abs(f_next) <= tol_final:
            return x_next
            
        if abs(x_next) > 1e10 * max(1.0, abs(x0)):
            break

        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_next, f_next
        
    if abs(f_curr) <= tol_final:
        return x_curr
    if abs(f_prev) <= tol_final:
        return x_prev

    # --- Phase 3: Bracketing Search ---
    a, fa = x_prev, f_prev
    b, fb = x_curr, f_curr
    
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
        
    bracket_found = (fa * fb < 0)
    
    if not bracket_found:
        # Try to find a bracket by expanding
        step_size = max(abs(b - a), 1e-4 * max(1.0, abs(a)))
        
        for i in range(15):
            if time.time() > deadline:
                break
            # Expand b
            b_new = b + step_size * (i + 1)
            fb_new = f(b_new)
            if math.isfinite(fb_new):
                if fa * fb_new < 0:
                    b, fb = b_new, fb_new
                    bracket_found = True
                    break
                if abs(fb_new) < best_f:
                    best_x, best_f = b_new, abs(fb_new)
                if abs(fb_new) <= tol_final:
                    return b_new
            # Expand a
            a_new = a - step_size * (i + 1)
            fa_new = f(a_new)
            if math.isfinite(fa_new):
                if fa_new * fb < 0:
                    a, fa = a_new, fa_new
                    bracket_found = True
                    break
                if abs(fa_new) < best_f:
                    best_x, best_f = a_new, abs(fa_new)
                if abs(fa_new) <= tol_final:
                    return a_new

    if not bracket_found:
        return best_x

    # --- Phase 4: Hybrid Bisection / IQI ---
    m = (a + b) / 2.0
    fm = f(m)
    
    if not math.isfinite(fm):
        # Pure bisection
        for _ in range(80):
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

    points = [(a, fa), (m, fm), (b, fb)]
    points.sort(key=lambda p: p[0])
    x1, f1 = points[0]
    x2, f2 = points[1]
    x3, f3 = points[2]
    
    if f1 * f2 < 0:
        x_lo, f_lo, x_hi, f_hi = x1, f1, x2, f2
        x_mid, f_mid = x3, f3
    elif f2 * f3 < 0:
        x_lo, f_lo, x_hi, f_hi = x2, f2, x3, f3
        x_mid, f_mid = x1, f1
    else:
        return best_x

    max_hybrid_iter = 80
    for _ in range(max_hybrid_iter):
        if time.time() > deadline:
            break
            
        if abs(f_lo) <= tol_final:
            return x_lo
        if abs(f_hi) <= tol_final:
            return x_hi
        if abs(f_mid) <= tol_final:
            return x_mid
            
        if abs(x_hi - x_lo) < 1e-15:
            return (x_lo + x_hi) / 2.0

        # IQI
        d1 = (f_lo - f_mid) * (f_lo - f_hi)
        d2 = (f_mid - f_lo) * (f_mid - f_hi)
        d3 = (f_hi - f_lo) * (f_hi - f_mid)
        
        x_iqi = None
        if abs(d1) > 1e-30 and abs(d2) > 1e-30 and abs(d3) > 1e-30:
            x_iqi = (x_lo * f_mid * f_hi / d1 +
                     x_mid * f_lo * f_hi / d2 +
                     x_hi * f_lo * f_mid / d3)
        
        margin = 1e-12 * max(1.0, abs(x_hi - x_lo))
        if (x_iqi is not None and math.isfinite(x_iqi) and 
            (x_lo + margin < x_iqi < x_hi - margin)):
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
            
        if abs(x_hi - x_lo) < 1e-10:
            break

    candidates = [(x_lo, f_lo), (x_hi, f_hi), (x_mid, f_mid), (best_x, best_f)]
    best_final = min(candidates, key=lambda p: abs(p[1]))
    return best_final[0]

# EVOLVE-BLOCK-END
