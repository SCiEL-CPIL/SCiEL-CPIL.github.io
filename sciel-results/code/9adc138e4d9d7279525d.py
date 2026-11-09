import numpy as np

N = 400
h = 1.0 / (2 * N)  # interval width: total length 1/2, so h = (1/2)/400 = 1/800

def autocorr_peak(f):
    """Compute max |autocorrelation| for step function f on [-1/4, 1/4]."""
    # f is array of length N, values at each interval
    # Autocorrelation R(k) = h * sum_i f[i] * f[i+k] for k from -(N-1) to (N-1)
    # We compute for all lags
    R = np.correlate(f, f, mode='full') * h
    return np.max(np.abs(R))

def construct():
    best_f = np.ones(N)
    best_peak = autocorr_peak(best_f)
    
    # Try alternating
    f_alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    peak_alt = autocorr_peak(f_alt)
    if peak_alt < best_peak:
        best_f = f_alt
        best_peak = peak_alt
    
    # Try random sequences
    rng = np.random.default_rng(42)
    for _ in range(5000):
        f_rand = rng.choice([-1.0, 1.0], size=N)
        peak = autocorr_peak(f_rand)
        if peak < best_peak:
            best_f = f_rand
            best_peak = peak
    
    return best_f
