"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_acf_peak(f):
    """Compute the maximum absolute value of the autocorrelation for lags 1..N-1."""
    # f is a numpy array of ±1 values of length N
    # acf[k] = sum_{i=0}^{N-1-k} f[i]*f[i+k]
    # We want max |acf[k]| for k in 1..N-1
    acf = np.correlate(f, f, mode='full')
    # acf has length 2N-1, centered at index N-1
    # acf[N-1+k] corresponds to lag k
    # acf[N-1-k] corresponds to lag -k
    # We want lags 1 to N-1
    peak = np.max(np.abs(acf[N:N-1]))  # lags 1 to N-1
    return peak

def _try_flip(f, i, current_peak):
    """Try flipping sign at position i, return (new_peak, new_f) if improved."""
    f_new = f.copy()
    f_new[i] = -f_new[i]
    new_peak = _compute_acf_peak(f_new)
    return new_peak, f_new

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    best_f = np.ones(N, dtype=np.float64)
    best_peak = _compute_acf_peak(best_f)
    
    # Try multiple random initializations
    np.random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        f = np.random.choice([-1.0, 1.0], size=N)
        peak = _compute_acf_peak(f)
        if peak < best_peak:
            best_peak = peak
            best_f = f.copy()
    
    # Local search: try flipping each position
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            new_peak, new_f = _try_flip(best_f, i, best_peak)
            if new_peak < best_peak - 1e-12:
                best_peak = new_peak
                best_f = new_f
                improved = True
    
    return best_f
# EVOLVE-BLOCK-END

