import numpy as np
import time

N = 400
_DEADLINE_SECONDS = 140  # safety margin before 150s kill

def _autocorr_peak(f):
    """Compute peak absolute autocorrelation for lags 1..N-1 using FFT."""
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    return np.max(np.abs(acf[1:]))

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as ±1 float array."""
    bits = np.array([bin(i).count('1') % 2 for i in range(n)])
    return 1.0 - 2.0 * bits

def _local_search(f_init, start_time, deadline):
    """Greedy local search: flip individual bits to reduce autocorrelation peak."""
    f = f_init.copy().astype(np.float64)
    best_f = f.copy()
    best_peak = _autocorr_peak(f)
    
    for pass_num in range(100):
        if time.time() - start_time > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() - start_time > deadline:
                break
            f[i] = -f[i]
            peak = _autocorr_peak(f)
            if peak < best_peak - 1e-12:
                best_peak = peak
                best_f = f.copy()
                improved = True
            else:
                f[i] = -f[i]  # revert
        if not improved:
            break
    
    return best_f, best_peak

def construct():
    start = time.time()
    deadline = start + _DEADLINE_SECONDS
    
    # Initialize with Thue-Morse sequence
    f0 = _thue_morse(N)
    best_f, best_peak = _local_search(f0, start, deadline)
    
    # Try random restarts if time permits
    rng = np.random.RandomState(42)
    for trial in range(30):
        if time.time() > deadline:
            break
        f_rand = rng.choice([-1.0, 1.0], size=N)
        f_res, peak_res = _local_search(f_rand, start, deadline)
        if peak_res < best_peak:
            best_f = f_res
            best_peak = peak_res
    
    return best_f
