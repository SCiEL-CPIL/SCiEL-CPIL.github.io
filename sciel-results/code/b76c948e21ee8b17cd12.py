import numpy as np
import time

N = 400

def _autocorr(x):
    """Compute autocorrelation of x."""
    n = len(x)
    ac = np.zeros(2*n - 1)
    # Use FFT for efficiency
    fft_size = 1
    while fft_size < 2*n - 1:
        fft_size *= 2
    X = np.fft.fft(x, fft_size)
    ac_full = np.fft.ifft(X * np.conj(X)).real[:2*n-1]
    # ac_full[k] for k=0..n-1 is autocorr at lag k
    # For k=n..2n-2, it's the negative lags
    ac = ac_full[:n]
    return ac

def _peak_ac(x):
    """Return max absolute autocorrelation value."""
    ac = _autocorr(x)
    return np.max(np.abs(ac))

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Start with a random ±1 sequence
    x = np.random.choice([-1, 1], size=N)
    best_x = x.copy()
    best_peak = _peak_ac(best_x)
    
    # Local search: flip one element at a time
    np.random.seed(42)
    for _ in range(10000):
        if time.time() > deadline - 5:
            break
        # Try flipping a random position
        idx = np.random.randint(N)
        old_val = x[idx]
        x[idx] = -old_val
        peak = _peak_ac(x)
        if peak < best_peak:
            best_peak = peak
            best_x = x.copy()
        else:
            x[idx] = old_val
    
    # Also try a greedy construction
    # Start with +1 and build up
    x_greedy = np.ones(N)
    for i in range(1, N):
        # Try both signs for position i
        x_greedy[i] = 1
        peak1 = _peak_ac(x_greedy)
        x_greedy[i] = -1
        peak2 = _peak_ac(x_greedy)
        if peak2 < peak1:
            x_greedy[i] = -1
        else:
            x_greedy[i] = 1
        if time.time() > deadline - 10:
            break
    
    peak_greedy = _peak_ac(x_greedy)
    if peak_greedy < best_peak:
        best_x = x_greedy.copy()
        best_peak = peak_greedy
    
    return best_x
