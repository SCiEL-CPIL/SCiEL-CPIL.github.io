import numpy as np
import time

N = 400

def _compute_autoconv(f):
    """Compute full autoconvolution via FFT"""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(f, size)
    power = F * np.conj(F)
    auto = np.fft.irfft(power, size)[:2 * n - 1]
    return auto

def _update_autoconv(auto, f, i, delta):
    """
    Update autoconvolution when f[i] changes by delta.
    """
    n = len(f)
    left = np.zeros(2*n-1)
    left[i:i+n] = f
    right = np.zeros(2*n-1)
    right[:n-i] = f[i:]
    diag = np.zeros(2*n-1)
    if 0 <= 2*i < 2*n-1:
        diag[2*i] = 1.0
    return auto + delta * (left + right) + delta * delta * diag

def _score_from_auto(auto, s, exclude_lag0=True):
    """Compute score from autoconvolution and sum"""
    if abs(s) < 1e-12:
        return float('inf')
    if exclude_lag0:
        # Exclude lag 0 (index 0)
        max_abs = np.max(np.abs(auto[1:]))
    else:
        max_abs = np.max(np.abs(auto))
    return max_abs / (s * s)

def _spectral_shaping(f, deadline, rng):
    """
    Iteratively shape the spectrum to minimize side-lobes in autoconvolution.
    We keep the DC component (sum) high and flatten the rest of the spectrum.
    """
    current_f = f.copy()
    best_f = f.copy()
    best_auto = _compute_autoconv(best_f)
    best_s = np.sum(best_f)
    best_score = _score_from_auto(best_auto, best_s)
    
    size = 1
    while size < 2 * N - 1:
        size <<= 1
        
    for _ in range(500):
        if time.time() > deadline - 10:
            break
            
        F = np.fft.rfft(current_f, size)
        
        # Target: Flat spectrum for non-DC bins
        # Calculate the target magnitude for non-DC bins
        # We want to minimize max |C[k]| for k != 0.
        # C[k] is IFFT of |F|^2.
        # If |F|^2 is constant for k != 0, then C[k] for k != 0 will be small (sinc-like).
        
        # Strategy: Set |F[1:]| to a small constant value, keep phase.
        # What constant? Maybe the RMS of |F[1:]| or just a fixed small value.
        # Let's try to reduce the peaks in |F|^2.
        
        mag = np.abs(F)
        phase = np.angle(F)
        
        # DC component (index 0) corresponds to sum S. We want to keep it high.
        # Non-DC components: we want to minimize their contribution to peaks in C.
        
        # Heuristic: Set non-DC magnitudes to a small fraction of the DC magnitude.
        # Or, more simply, smooth the magnitude spectrum.
        
        # Let's try: |F_new[k]| = sqrt(|F[k]|^2) for k>0? No.
        # Let's try: |F_new[k]| = constant for k>0.
        # What constant? If we set it too low, the IFFT will have large negative values, 
        # and rounding to +/-1 might flip many signs, reducing S.
        
        # Better approach: 
        # 1. Compute current |F|^2.
        # 2. Identify peaks in |F|^2 (excluding DC).
        # 3. Reduce those peaks.
        
        # Simpler: 
        # Set |F[1:]| to the median of |F[1:]|.
        target_mag = np.median(mag[1:])
        
        new_mag = mag.copy()
        new_mag[1:] = target_mag
        
        new_F = new_mag * np.exp(1j * phase)
        
        # Inverse FFT
        new_f = np.fft.irfft(new_F, size)
        
        # Round to +/- 1
        # Use sign, but handle 0s by keeping previous value or random
        new_f_rounded = np.sign(new_f[:N])
        # If 0, keep old value
        zero_mask = (new_f_rounded == 0)
        new_f_rounded[zero_mask] = current_f[zero_mask]
        
        # Update current_f
        current_f = new_f_rounded
        
        # Evaluate
        auto = _compute_autoconv(current_f)
        s = np.sum(current_f)
        score = _score_from_auto(auto, s)
        
        if score < best_score:
            best_score = score
            best_f = current_f.copy()
            best_auto = auto.copy()
            best_s = s
            
    return best_f, best_auto, best_s, best_score

def _local_search(f, auto, s, current_score, best_score, best_f, best_auto, best_s, deadline, rng):
    """Greedy local search with incremental autoconv update."""
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        order = rng.permutation(N)
        for i in order:
            if time.time() > deadline - 5:
                break
            old_val = f[i]
            delta = -2 * old_val
            new_auto = _update_autoconv(auto, f, i, delta)
            new_s = s + delta
            new_score = _score_from_auto(new_auto, new_s)
            if new_score < current_score - 1e-12:
                f[i] = old_val + delta
                auto = new_auto
                s = new_s
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
                    best_auto = auto.copy()
                    best_s = s
                improved = True
    return f, auto, s, current_score, best_score, best_f, best_auto, best_s

def construct():
    deadline = time.time() + 115
    
    # Start with all +1s
    best_f = np.ones(N)
    best_auto = _compute_autoconv(best_f)
    best_s = np.sum(best_f)
    best_score = _score_from_auto(best_auto, best_s)
    
    # Spectral Shaping
    rng = np.random.default_rng(42)
    best_f, best_auto, best_s, best_score = _spectral_shaping(best_f, deadline, rng)
    
    # Local Search
    if time.time() < deadline - 5:
        f = best_f.copy()
        auto = best_auto.copy()
        s = best_s
        current_score = best_score
        rng = np.random.default_rng(123)
        
        f, auto, s, current_score, best_score, best_f, best_auto, best_s = _local_search(
            f, auto, s, current_score, best_score, best_f, best_auto, best_s, deadline, rng)
            
    return best_f
