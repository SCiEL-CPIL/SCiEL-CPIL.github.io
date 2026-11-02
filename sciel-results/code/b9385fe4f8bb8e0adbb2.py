import numpy as np
import time

N = 400

def _compute_c3(f):
    """Compute third-order autocorrelation for all lags.
    f: array of length N with values ±1.
    Returns array c3[k] = sum_i f[i]*f[i+k]*f[i+2k] for k=0..N//2-1 (or up to where valid).
    """
    n = len(f)
    max_lag = n // 2
    c3 = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        # f[i], f[i+k], f[i+2k] for i = 0..n-1-2k
        n_terms = n - 2*k
        if n_terms <= 0:
            break
        c3[k] = np.sum(f[:n_terms] * f[k:k+n_terms] * f[2*k:2*k+n_terms])
    return c3

def _score(f):
    """Return the metric value (lower is better)."""
    c3 = _compute_c3(f)
    # Normalize: for f=1, c3[0] = N, and we want score=1.0.
    # If the metric is max|c3| / N, then for f=1 it's N/N = 1.0.
    return np.max(np.abs(c3)) / N

def construct():
    start = time.time()
    deadline = 110.0  # safety margin
    
    # Start with a random sequence
    rng = np.random.default_rng(42)
    f = rng.choice([-1, 1], size=N).astype(np.float64)
    best_f = f.copy()
    best_val = _score(f)
    
    # Simulated annealing
    T = 0.1
    T_min = 1e-6
    cooling = 0.999
    
    while time.time() - start < deadline:
        # Pick a random position to flip
        i = rng.integers(0, N)
        old = f[i]
        f[i] = -f[i]
        new_val = _score(f)
        
        delta = new_val - best_val
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Accept
            if new_val < best_val:
                best_val = new_val
                best_f = f.copy()
        else:
            # Reject
            f[i] = old
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_f
