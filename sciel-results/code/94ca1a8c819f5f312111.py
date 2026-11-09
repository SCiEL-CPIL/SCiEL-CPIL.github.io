"""DFT-based spectral root-finder with Newton refinement."""
import math
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Strategy:
      1. Sample f at a small number of equally-spaced points around x0.
      2. Use a DFT to estimate the dominant frequency and phase of the oscillation.
      3. Predict the nearest zero-crossing from the fitted sinusoid.
      4. Refine with Newton's method (finite-difference derivative), with bisection fallback.
    """
    x0 = float(x0)
    
    # --- Step 1: Initial evaluation ---
    best_x = x0
    best_fx = f(x0)
    best_abs = abs(best_fx)
    
    if best_abs <= 1e-8:
        return best_x
    
    # --- Step 2: Adaptive window selection and DFT-based frequency estimation ---
    N = 32  # number of samples for DFT (increased for better frequency resolution)
    
    # Try multiple window sizes to capture the oscillation
    base_window = max(0.5, abs(x0) * 0.05, 0.05)
    windows = [base_window * s for s in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]]
    windows = [w for w in windows if w > 0]
    
    best_candidate_x = best_x
    best_candidate_fx = best_fx
    best_candidate_abs = best_abs
    
    for window in windows:
        # Sample N+1 points over [-window, window] centered at x0
        xs = x0 + np.linspace(-window, window, N + 1)
        xs = np.unique(xs)
        if len(xs) < 8:
            continue
        N_actual = len(xs)
        
        try:
            fs = np.array([f(float(x)) for x in xs])
        except Exception:
            continue
        
        if not np.all(np.isfinite(fs)):
            continue
        
        # Track best so far from samples
        for xi, fi in zip(xs, fs):
            if abs(fi) < best_candidate_abs:
                best_candidate_abs = abs(fi)
                best_candidate_x = float(xi)
                best_candidate_fx = float(fi)
        
        if best_candidate_abs <= 1e-8:
            return best_candidate_x
        
        # --- DFT analysis ---
        f_mean = np.mean(fs)
        fs_centered = fs - f_mean
        
        # Compute DFT
        F = np.fft.fft(fs_centered)
        freqs = np.fft.fftfreq(N_actual, d=(2 * window) / (N_actual - 1))
        magnitudes = np.abs(F)
        magnitudes[0] = 0  # Exclude DC
        
        if len(magnitudes) < 2:
            continue
        
        # Find top 3 frequency components (excluding DC)
        sorted_indices = np.argsort(magnitudes)[::-1]
        sorted_indices = sorted_indices[sorted_indices > 0]  # Exclude DC
        if len(sorted_indices) < 1:
            continue
        
        # Try the top 3 frequencies
        for idx in sorted_indices[:3]:
            if magnitudes[idx] < 1e-10:
                continue
            dominant_freq = abs(freqs[idx])
            if dominant_freq < 1e-10:
                continue
            
            # Amplitude and phase from DFT
            A = 2.0 * magnitudes[idx] / N_actual
            phase = np.angle(F[idx])
            
            if abs(A) < 1e-10:
                continue
            
            # The fitted sinusoid: A * cos(2*pi*f*(x - x0) + phi) + f_mean = 0
            # cos(2*pi*f*(x-x0) + phi) = -f_mean / A
            ratio = -f_mean / A
            if ratio < -1.0:
                ratio = -1.0
            elif ratio > 1.0:
                ratio = 1.0
            
            theta = np.arccos(ratio)
            
            # Two families of zero-crossings
            candidates = []
            for m in range(-4, 5):
                x1 = x0 + (theta - phase + 2 * math.pi * m) / (2 * math.pi * dominant_freq)
                x2 = x0 + (-theta - phase + 2 * math.pi * m) / (2 * math.pi * dominant_freq)
                candidates.append(float(x1))
                candidates.append(float(x2))
            
            # Evaluate candidates
            for xc in candidates:
                if not np.isfinite(xc):
                    continue
                try:
                    fxc = f(xc)
                except Exception:
                    continue
                if not np.isfinite(fxc):
                    continue
                if abs(fxc) < best_candidate_abs:
                    best_candidate_abs = abs(fxc)
                    best_candidate_x = xc
                    best_candidate_fx = fxc
            
            if best_candidate_abs <= 1e-8:
                return best_candidate_x
        
        # If we found a significantly better candidate, we can stop
        if best_candidate_abs < best_abs * 0.1:
            break
    
    # --- Step 3: Newton refinement with bisection fallback ---
    x = float(best_candidate_x)
    fx = best_candidate_fx
    
    # First, try to establish a bracket around the root using sign changes
    bracket_lo = None
    bracket_hi = None
    h_bracket = max(0.01, abs(x) * 0.01)
    
    # Search for a sign change nearby
    for step in [h_bracket, h_bracket * 0.5, h_bracket * 2, h_bracket * 4, h_bracket * 0.1]:
        try:
            f_lo = f(x - step)
            f_hi = f(x + step)
        except Exception:
            continue
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
            continue
        if f_lo * f_hi <= 0:
            bracket_lo = x - step
            bracket_hi = x + step
            break
    
    # Newton iteration with damping and bisection fallback
    for _ in range(50):
        if abs(fx) <= 1e-8:
            return x
        
        # Compute derivative via central difference
        h = 1e-6 * max(1.0, abs(x))
        try:
            fxp = f(x + h)
            fxm = f(x - h)
        except Exception:
            break
        if not (np.isfinite(fxp) and np.isfinite(fxm)):
            break
        d = (fxp - fxm) / (2 * h)
        
        if d == 0 or not np.isfinite(d):
            # Fall back to bisection if we have a bracket
            if bracket_lo is not None and bracket_hi is not None:
                mid = 0.5 * (bracket_lo + bracket_hi)
                try:
                    f_mid = f(mid)
                except Exception:
                    break
                if not np.isfinite(f_mid):
                    break
                if abs(f_mid) <= 1e-8:
                    return mid
                if f_mid * f(bracket_lo) <= 0:
                    bracket_hi = mid
                else:
                    bracket_lo = mid
                x = mid
                fx = f_mid
                continue
            break
        
        # Newton step
        x_new = x - fx / d
        
        # Damping: if step is too large, reduce it
        step = abs(x_new - x)
        max_step = max(0.1, abs(x) * 0.5)
        if step > max_step:
            x_new = x + math.copysign(max_step, x_new - x)
        
        if not np.isfinite(x_new):
            break
        
        try:
            fx_new = f(x_new)
        except Exception:
            break
        if not np.isfinite(fx_new):
            break
        
        # If we have a bracket, check if Newton step stays within it
        if bracket_lo is not None and bracket_hi is not None:
            if fx_new * fx <= 0:
                # Sign change: update bracket
                if fx_new * f(bracket_lo) <= 0:
                    bracket_hi = x_new
                else:
                    bracket_lo = x_new
            else:
                # No sign change: use bisection instead
                mid = 0.5 * (bracket_lo + bracket_hi)
                try:
                    f_mid = f(mid)
                except Exception:
                    break
                if not np.isfinite(f_mid):
                    break
                if abs(f_mid) <= 1e-8:
                    return mid
                if f_mid * f(bracket_lo) <= 0:
                    bracket_hi = mid
                else:
                    bracket_lo = mid
                x = mid
                fx = f_mid
                continue
        
        x = x_new
        fx = fx_new
    
    return x
# EVOLVE-BLOCK-END
