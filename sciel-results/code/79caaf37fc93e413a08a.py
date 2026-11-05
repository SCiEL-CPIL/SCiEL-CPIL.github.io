"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _autoconvolution_peak(x):
    """Compute the peak absolute value of the linear autoconvolution."""
    n = len(x)
    # Linear autoconvolution via FFT (zero-padded to avoid circular wrap)
    fft_len = 2 * n
    X = np.fft.rfft(x, n=fft_len)
    conv = np.fft.irfft(X * X, n=fft_len)
    # The linear convolution of x with x has length 2n-1
    # conv[0] corresponds to the full correlation at lag 0 (sum of x^2 = n for +/-1)
    # We want the peak over all lags. For autoconvolution, we consider the full result.
    # Actually, np.correlate(x, x, 'full') gives the same as conv[0:n] + conv[n+1:2n-1] reversed?
    # Let's just use the direct definition or a careful FFT.
    # For x in {-1, 1}, the autoconvolution at lag k is sum_{i} x[i] * x[i+k].
    # The peak is max_k |sum_i x[i] x[i+k]|.
    # Using FFT: ifft(fft(x, 2n)^2) gives the circular convolution of x with x padded to 2n.
    # This equals the linear convolution because x is zero outside [0, n-1].
    # The result has length 2n-1. The FFT result has length 2n.
    # The linear convolution c[k] for k=0..2n-2 is in conv[0..2n-2].
    # But wait, the FFT convolution of two length-n sequences padded to 2n gives a length-2n result
    # where the first n values are the "direct" part and the last n are the "wrapped" part?
    # No, for linear convolution of two length-n sequences, we need FFT size >= 2n-1.
    # Using size 2n is sufficient. The result is the linear convolution in positions 0 to 2n-2.
    # Position 2n-1 is zero (or numerical noise).
    return np.max(np.abs(conv[:2*n-1]))


def _smooth_peak_surrogate(x, temperature=1.0):
    """
    Smooth surrogate for the peak autoconvolution using log-sum-exp.
    We want to minimize max_k |conv[k]|.
    This is equivalent to minimizing the soft-max of |conv[k]|.
    soft_max(z, T) = T * log(sum(exp(z_k / T)))
    Here z_k = |conv[k]|.
    """
    n = len(x)
    fft_len = 2 * n
    X = np.fft.rfft(x, n=fft_len)
    conv = np.fft.irfft(X * X, n=fft_len)
    conv = conv[:2*n-1]
    abs_conv = np.abs(conv)
    
    # To avoid overflow in exp, subtract max
    max_val = np.max(abs_conv)
    if max_val == 0:
        return 0.0
    
    # Use a stable log-sum-exp
    # log(sum(exp(z/T))) = max_z/T + log(sum(exp((z - max_z)/T)))
    scaled = abs_conv / temperature
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return temperature * log_sum_exp


def _local_search(x, time_limit=20):
    """Local search: flip individual bits to reduce autoconvolution peak."""
    x = x.copy()
    best_peak = _autoconvolution_peak(x)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        n = len(x)
        # Shuffle order to avoid bias
        indices = np.random.permutation(n)
        for i in indices:
            if (time.time() - start_time) > time_limit:
                break
            # Try flipping position i
            x[i] *= -1
            new_peak = _autoconvolution_peak(x)
            if new_peak < best_peak - 1e-12:
                best_peak = new_peak
                improved = True
            else:
                x[i] *= -1  # revert
    return x, best_peak


def _pairwise_flip_search(x, time_limit=20):
    """Pairwise flip search: flip two bits simultaneously to escape local minima."""
    x = x.copy()
    best_peak = _autoconvolution_peak(x)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        n = len(x)
        # Try a few random pairs
        for _ in range(100):
            if (time.time() - start_time) > time_limit:
                break
            i, j = np.random.choice(n, size=2, replace=False)
            x[i] *= -1
            x[j] *= -1
            new_peak = _autoconvolution_peak(x)
            if new_peak < best_peak - 1e-12:
                best_peak = new_peak
                improved = True
            else:
                x[i] *= -1
                x[j] *= -1
    return x, best_peak


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n, mapped to +/-1."""
    seq = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # Count bits
        c = 0
        val = i
        while val:
            c += val & 1
            val >>= 1
        seq[i] = 1 if c % 2 == 0 else -1
    return seq.astype(np.float64)


def _rule90_sequence(seed_pos, k_steps, total_len):
    """Generate Rule 90 CA sequence from a single seed."""
    row = np.zeros(total_len, dtype=np.uint8)
    row[seed_pos] = 1
    
    for _ in range(k_steps):
        left = np.roll(row, 1)
        right = np.roll(row, -1)
        new_row = (left ^ right) & 1
        row = new_row
    
    return 2 * row.astype(np.float64) - 1


def construct():
    start_time = time.time()
    deadline = 150.0  # Evaluator kills at 150s, keep safety margin
    
    best_x = None
    best_peak = np.inf
    
    # Initial candidates: Thue-Morse and Rule 90
    candidates = []
    
    # Thue-Morse
    candidates.append(_thue_morse(N))
    
    # Rule 90 with various seeds
    for seed_pos in [0, N//2, N//4, 3*N//4]:
        for k_steps in [10, 50, 100, 200]:
            candidates.append(_rule90_sequence(seed_pos, k_steps, N))
    
    # Random initialization
    candidates.append(np.random.choice([-1.0, 1.0], size=N))
    
    # Local search on each candidate
    for x in candidates:
        if (time.time() - start_time) > deadline - 30:
            break
        
        x_refined, peak = _local_search(x, time_limit=10)
        if peak < best_peak:
            best_peak = peak
            best_x = x_refined
    
    # Pairwise flip search on the best candidate
    if best_x is not None and (time.time() - start_time) < deadline - 20:
        x_pair, peak_pair = _pairwise_flip_search(best_x, time_limit=15)
        if peak_pair < best_peak:
            best_peak = peak_pair
            best_x = x_pair
    
    # Final local search polish
    if best_x is not None and (time.time() - start_time) < deadline - 10:
        x_final, peak_final = _local_search(best_x, time_limit=10)
        if peak_final < best_peak:
            best_peak = peak_final
            best_x = x_final
    
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
