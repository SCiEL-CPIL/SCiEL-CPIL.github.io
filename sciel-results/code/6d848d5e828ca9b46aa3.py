import numpy as np
import time

N = 400

def _autoconv_peak_fft(signs):
    """Compute max |autocorr(k)| for k=1..N-1 using FFT."""
    n = len(signs)
    size = 2 * n
    f = np.fft.fft(signs, size=size)
    power = f * np.conj(f)
    ac = np.fft.ifft(power).real
    if n > 1:
        return np.max(np.abs(ac[1:n]))
    return 0.0

def _generate_fbM_signs(N, H, seed=42):
    """Generate ±1 signs from a process with fBm-like power spectrum."""
    rng = np.random.RandomState(seed)
    x = rng.randn(N)
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N)
    
    mask = freqs != 0
    filt = np.ones(N)
    filt[mask] = np.abs(freqs[mask]) ** (-(2*H + 1))
    
    Y = X * np.sqrt(filt)
    y = np.fft.ifft(Y).real
    y = (y - y.mean()) / (y.std() + 1e-10)
    signs = np.sign(y)
    signs[signs == 0] = 1.0
    return signs

def _local_search(signs, time_limit=5.0, max_iters=200000):
    """Greedy local search with FFT-based evaluation."""
    n = len(signs)
    signs = signs.copy()
    current_peak = _autoconv_peak_fft(signs)
    best_peak = current_peak
    best_signs = signs.copy()
    start = time.time()
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        improved = False
        for idx in range(n):
            old = signs[idx]
            signs[idx] = -old
            new_peak = _autoconv_peak_fft(signs)
            if new_peak < current_peak - 1e-12:
                current_peak = new_peak
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_signs = signs.copy()
                improved = True
                break
            signs[idx] = old
        
        if not improved:
            break
    
    return best_signs, best_peak

def construct():
    start_time = time.time()
    deadline = start_time + 105.0
    
    best_signs = np.ones(N)
    best_peak = _autoconv_peak_fft(best_signs)
    
    H_values = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
    
    for H in H_values:
        if time.time() > deadline:
            break
        for seed in range(3):
            if time.time() > deadline:
                break
            signs = _generate_fbM_signs(N, H, seed=seed + 100)
            refined, peak = _local_search(signs, time_limit=8.0)
            if peak < best_peak:
                best_peak = peak
                best_signs = refined.copy()
    
    if time.time() < deadline - 3.0:
        final, fp = _local_search(best_signs, time_limit=10.0)
        if fp < best_peak:
            best_signs = final
    
    return best_signs
