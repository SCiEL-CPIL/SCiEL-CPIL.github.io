import numpy as np
import time
from scipy import signal

N = 400

def _autoconv_max_abs_nonzero(x):
    """Compute max absolute value of full autoconvolution of x, excluding lag 0."""
    n = len(x)
    size = 2 * n
    fft_x = np.fft.fft(x, size)
    acov = np.fft.ifft(fft_x * np.conj(fft_x)).real
    # acov[0] is lag 0. We want max of acov[1:n] and acov[n:2n-1] (negative lags)
    # Due to symmetry, max(abs(acov[1:n])) is sufficient
    if n <= 1:
        return 0.0
    return np.max(np.abs(acov[1:n]))

def _score_nonzero(x):
    """Compute the C3' ratio: max|autoconv_nonzero| / (sum(x))^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs_nonzero(x) / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_x = np.ones(N)
    best_val = _score_nonzero(best_x)
    
    # Try various windows
    # Hamming
    x_hamming = signal.get_window('hamming', N)
    val_hamming = _score_nonzero(x_hamming)
    if val_hamming < best_val:
        best_val = val_hamming
        best_x = x_hamming.copy()
        
    # Hanning
    x_hanning = signal.get_window('hann', N)
    val_hanning = _score_nonzero(x_hanning)
    if val_hanning < best_val:
        best_val = val_hanning
        best_x = x_hanning.copy()
        
    # Blackman
    x_blackman = signal.get_window('blackman', N)
    val_blackman = _score_nonzero(x_blackman)
    if val_blackman < best_val:
        best_val = val_blackman
        best_x = x_blackman.copy()
        
    # Kaiser with various beta
    for beta in [3.0, 5.0, 7.0, 9.0, 12.0]:
        if time.time() > deadline - 10:
            break
        x_kaiser = signal.get_window('kaiser', N, beta=beta)
        val_kaiser = _score_nonzero(x_kaiser)
        if val_kaiser < best_val:
            best_val = val_kaiser
            best_x = x_kaiser.copy()

    # Local search on the best candidate to further reduce sidelobes
    # Simple gradient-free local search on real values
    x_curr = best_x.copy()
    curr_val = best_val
    rng = np.random.RandomState(42)
    
    step_size = 0.1
    for it in range(10000):
        if time.time() > deadline - 5:
            break
        idx = rng.randint(N)
        x_trial = x_curr.copy()
        x_trial[idx] += rng.uniform(-step_size, step_size)
        # Ensure non-negative? "Signed step heights" might allow negative, 
        # but windows are positive. If we allow negative, sum might drop.
        # Let's stick to positive for now to keep sum large.
        trial_val = _score_nonzero(x_trial)
        if trial_val < curr_val:
            x_curr = x_trial
            curr_val = trial_val
            
    return x_curr

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
