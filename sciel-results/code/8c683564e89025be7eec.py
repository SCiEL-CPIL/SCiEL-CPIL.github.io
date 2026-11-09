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
      4. Refine with Newton's method (finite-difference derivative).
    """
    x0 = float(x0)
    
    best_x = x0
    best_fx = f(x0)
    best_abs = abs(best_fx)
    
    if best_abs <= 1e-8:
        return best_x
    
    # Try multiple window sizes to capture the oscillation period
    # Use a smaller N to reduce call count, but try more windows
    N = 12  # number of samples for DFT (reduced from 16)
    
    base_window = max(0.1, abs(x0) * 0.05, 0.05)
    
    # Try several window scales - prioritize smaller windows for local roots
    window_factors = [1.0, 0.5, 2.0, 0.25, 4.0, 0.125, 8.0, 0.0625]
    
    for factor in window_factors:
        window = base_window * factor
        if window <= 0 or not np.isfinite(window):
            continue
        
        # Sample N+1 points over [-window, window] centered at x0
        xs = x0 + np.linspace(-window, window, N + 1)
        # Remove duplicates
        xs = np.unique(xs)
        if len(xs) < 4:
            continue
        N_actual = len(xs)
        
        try:
            fs = np.array([f(float(x)) for x in xs])
        except Exception:
            continue
        
        if not np.all(np.isfinite(fs)):
            continue
        
        # Track best so far
        for xi, fi in zip(xs, fs):
            if abs(fi) < best_abs:
                best_abs = abs(fi)
                best_x = float(xi)
                best_fx = float(fi)
        
        if best_abs <= 1e-8:
            return best_x
        
        # --- DFT to estimate dominant frequency and phase ---
        f_mean = np.mean(fs)
        fs_centered = fs - f_mean
        
        # Check if there's significant oscillation
        if np.std(fs_centered) < 1e-10:
            continue
        
        F = np.fft.fft(fs_centered)
        freqs = np.fft.fftfreq(N_actual, d=(2 * window) / (N_actual - 1))
        
        magnitudes = np.abs(F)
        magnitudes[0] = 0
        if len(magnitudes) < 2:
            continue
        
        # Find top 2 dominant frequencies to handle potential aliasing
        idxs = np.argsort(magnitudes)[::-1]
        
        for idx in idxs[:3]:  # Try top 3 frequencies
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
            
            # Solve A*cos(2*pi*f*(x-x0) + phi) + f_mean = 0
            ratio = -f_mean / A
            if ratio < -1.0:
                ratio = -1.0
            elif ratio > 1.0:
                ratio = 1.0
            
            theta = np.arccos(ratio)
            
            # Generate candidates from both solution families
            candidates = []
            for m in range(-2, 3):
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
                if abs(fxc) < best_abs:
                    best_abs = abs(fxc)
                    best_x = xc
                    best_fx = fxc
            
            if best_abs <= 1e-8:
                return best_x
            
            # If we found a significantly better candidate, stop searching
            if best_abs < 1e-4:
                break
        
        # If we have a good enough candidate, break to Newton
        if best_abs < 1e-3:
            break
    
    # --- Newton refinement with bisection fallback ---
    x = float(best_x)
    fx = best_fx
    
    # First, try to establish a bracket if we don't have one
    # Use the samples we already have to find a sign change
    bracket_lo = None
    bracket_hi = None
    
    # Search for a bracket near x
    search_radius = max(0.1, abs(x) * 0.1)
    n_bracket_search = 20
    bracket_xs = np.linspace(x - search_radius, x + search_radius, n_bracket_search)
    bracket_fxs = []
    
    for bx in bracket_xs:
        try:
            bfx = f(float(bx))
        except Exception:
            bfx = float('nan')
        bracket_fxs.append(bfx)
    
    for i in range(len(bracket_xs) - 1):
        if not (np.isfinite(bracket_fxs[i]) and np.isfinite(bracket_fxs[i+1])):
            continue
        if bracket_fxs[i] * bracket_fxs[i+1] < 0:
            bracket_lo = float(bracket_xs[i])
            bracket_hi = float(bracket_xs[i+1])
            break
    
    # Newton with bisection safeguard
    for iteration in range(50):
        if abs(fx) <= 1e-8:
            return x
        
        h = 1e-7 * max(1.0, abs(x))
        try:
            fxp = f(x + h)
            fxm = f(x - h)
        except Exception:
            # Fall back to bisection if we have a bracket
            if bracket_lo is not None and bracket_hi is not None:
                x = (bracket_lo + bracket_hi) / 2
                try:
                    fx = f(x)
                except Exception:
                    break
                if not np.isfinite(fx):
                    break
                continue
            break
        
        if not (np.isfinite(fxp) and np.isfinite(fxm)):
            if bracket_lo is not None and bracket_hi is not None:
                x = (bracket_lo + bracket_hi) / 2
                try:
                    fx = f(x)
                except Exception:
                    break
                if not np.isfinite(fx):
                    break
                continue
            break
        
        d = (fxp - fxm) / (2 * h)
        if d == 0 or not np.isfinite(d):
            if bracket_lo is not None and bracket_hi is not None:
                x = (bracket_lo + bracket_hi) / 2
                try:
                    fx = f(x)
                except Exception:
                    break
                if not np.isfinite(fx):
                    break
                continue
            break
        
        x_new = x - fx / d
        
        if not np.isfinite(x_new):
            if bracket_lo is not None and bracket_hi is not None:
                x_new = (bracket_lo + bracket_hi) / 2
            else:
                break
        
        # Check if Newton step is within bracket
        if bracket_lo is not None and bracket_hi is not None:
            if x_new < min(bracket_lo, bracket_hi) or x_new > max(bracket_lo, bracket_hi):
                x_new = (bracket_lo + bracket_hi) / 2
        
        # Damping for large steps
        step = abs(x_new - x)
        if step > max(1.0, abs(x)) * 0.5:
            x_new = x + math.copysign(min(step, max(1.0, abs(x)) * 0.1), x_new - x)
        
        x = x_new
        try:
            fx = f(x)
        except Exception:
            break
        if not np.isfinite(fx):
            break
        
        # Update bracket if we have one
        if bracket_lo is not None and bracket_hi is not None:
            if fx * f(bracket_lo) <= 0:
                bracket_hi = x
            elif fx * f(bracket_hi) <= 0:
                bracket_lo = x
    
    return x
# EVOLVE-BLOCK-END
