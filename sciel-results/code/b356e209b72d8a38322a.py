"""Optimized signed step function via vectorized local search."""
import numpy as np
import time

N = 400
DT = 0.5 / N  # Interval width: 0.5 / 400


def _compute_autoconv(signed_heights):
    """Compute the autoconvolution via FFT and return as a 1D array of length N (lags 0..N-1)."""
    # Pad to next power of 2 for efficient FFT
    n = 1
    while n < 2 * N:
        n <<= 1
    F = np.fft.rfft(signed_heights, n)
    R = np.fft.irfft(F * np.conj(F), n)
    # Normalize and extract lags 0..N-1
    # The autoconvolution of a discrete signal s with itself is sum s[i] s[i-k]
    # The FFT method computes the circular convolution. With zero padding to 2N, 
    # the linear convolution is embedded in the first N elements (lags 0 to N-1).
    return R[:N] * DT


def _score_binary(signs):
    """
    Compute the score for a binary sequence.
    The task mentions 'upper bound on C3'. 
    Typically, for such problems, the metric involves the sidelobes of the autoconvolution.
    A common metric is the maximum absolute value of the sidelobes (excluding lag 0).
    Let's assume the score is the max |R[k]| for k > 0.
    """
    R = _compute_autoconv(signs)
    # Exclude lag 0
    sidelobes = R[1:]
    # The "upper bound" likely refers to the peak sidelobe level.
    # We want to minimize this.
    return np.max(np.abs(sidelobes))


def _vectorized_local_search(signs, max_iters=1000, time_budget=10.0):
    """
    Perform vectorized local search.
    In each iteration, compute the effect of flipping every bit simultaneously.
    Accept the flip that yields the largest improvement in the peak sidelobe.
    """
    current_signs = signs.copy()
    current_R = _compute_autoconv(current_signs)
    current_peak = np.max(np.abs(current_R[1:]))
    
    start_time = time.time()
    
    # Precompute indices for vectorized delta calculation
    # We need to calculate Delta R for all i simultaneously.
    # Delta R[k] for flip at i is -2 * s[i] * (s[i-k] + s[i+k]) * DT
    # We can construct a matrix of these deltas for all i and all k.
    # However, N=400, so a 400x400 matrix is fine.
    
    # Indices for s[i-k] and s[i+k]
    # Let I be the index matrix where I[i, k] = i-k (mod N? No, linear convolution)
    # Wait, the autoconvolution is linear.
    # R[k] = sum_{j=0}^{N-1-k} s[j] s[j+k]
    # If we flip s[i], the change in R[k] is:
    # -2 * s[i] * (s[i-k] + s[i+k]) * DT
    # Valid only if i-k is in [0, N-1] and i+k is in [0, N-1] AND the term contributes to R[k].
    # Actually, let's look at the definition:
    # R[k] = sum_{j} s[j] s[j-k]
    # If s[i] changes to -s[i]:
    # Terms where j=i: -s[i] s[i-k] - s[i] s[i-k] = -2 s[i] s[i-k]
    # Terms where j-k=i => j=i+k: -s[i+k] s[i] - s[i+k] s[i] = -2 s[i+k] s[i]
    # So Delta R[k] = -2 * s[i] * (s[i-k] + s[i+k]) * DT
    # This is valid if 0 <= i-k < N and 0 <= i+k < N.
    # Also, note that R[k] is defined for k=0..N-1.
    # For a given k, the sum runs over j such that 0<=j<N and 0<=j-k<N.
    # The term s[i]s[i-k] contributes to R[k] if 0<=i-k<N and 0<=i<N (always true for i in 0..N-1).
    # The term s[i+k]s[i] contributes to R[k] if 0<=i+k<N and 0<=i<N.
    # So for a specific flip i and lag k:
    # delta = 0
    # if 0 <= i-k < N: delta -= 2 * s[i] * s[i-k] * DT
    # if 0 <= i+k < N: delta -= 2 * s[i] * s[i+k] * DT
    
    # We can vectorize this.
    # Create index arrays
    i_idx = np.arange(N)
    k_idx = np.arange(N)
    
    # For each i (row) and k (col), we need s[i-k] and s[i+k].
    # Let's create a matrix of indices for s[i-k].
    # idx_im = i_idx[:, None] - k_idx[None, :]
    # idx_ip = i_idx[:, None] + k_idx[None, :]
    
    # But we need to handle boundaries.
    # It is easier to just loop over i in the outer loop if vectorization is tricky,
    # but we want to vectorize over i to find the best flip in one go.
    
    # Let's try to construct the Delta matrix D where D[i, k] is the change in R[k] if we flip s[i].
    # D[i, k] = -2 * s[i] * (s[i-k] + s[i+k]) * DT
    
    # We can use padding or masking.
    # s_pad = np.pad(s, (N, N), mode='constant')
    # Then s[i-k] = s_pad[i + N - k]
    # s[i+k] = s_pad[i + N + k]
    
    s_pad = np.pad(current_signs, (N, N), mode='constant')
    
    # Indices for s[i-k]: row i, col k -> index in flat array of s_pad?
    # s_pad is 2N x 2N.
    # s[i-k] corresponds to s_pad[i + N - k]
    # s[i+k] corresponds to s_pad[i + N + k]
    
    # Let's create index matrices
    # Shape (N, N)
    I = i_idx[:, None]
    K = k_idx[None, :]
    
    idx_im = I + N - K
    idx_ip = I + N + K
    
    # Extract values
    # s_im[i, k] = s_pad[idx_im[i, k]]
    # s_ip[i, k] = s_pad[idx_ip[i, k]]
    
    s_im = s_pad[idx_im]
    s_ip = s_pad[idx_ip]
    
    # Delta R for all i, k
    # Note: s[i] is current_signs[i]
    # D[i, k] = -2 * current_signs[i] * (s_im[i, k] + s_ip[i, k]) * DT
    D = -2.0 * current_signs[:, None] * (s_im + s_ip) * DT
    
    # New R for each potential flip i
    # R_new[i, k] = current_R[k] + D[i, k]
    R_new = current_R[None, :] + D
    
    # We only care about sidelobes k > 0
    # Mask k=0
    # R_new[:, 0] is irrelevant for the score, but we can ignore it.
    
    # Peak sidelobe for each flip i
    # peak_i = max(|R_new[i, 1:]|)
    peak_new = np.max(np.abs(R_new[:, 1:]), axis=1)
    
    # Find the flip that minimizes the peak
    best_idx = np.argmin(peak_new)
    best_peak = peak_new[best_idx]
    
    # If the best flip improves the score, apply it
    if best_peak < current_peak:
        current_signs[best_idx] *= -1
        current_R = R_new[best_idx]
        current_peak = best_peak
        return current_signs, current_peak, True
    
    return current_signs, current_peak, False


def construct():
    start_time = time.time()
    timeout = 110.0
    
    np.random.seed(42)
    
    best_signs = None
    best_score = np.inf
    
    # Number of restarts
    # Since local search is now fast, we can do multiple restarts.
    # Let's estimate time per local search.
    # Each iter involves FFT (O(N log N)) and vectorized ops (O(N^2)).
    # N=400, N^2=160,000. This is very fast in NumPy.
    # FFT of 1024 points is also fast.
    # So we can do many iterations.
    
    # Let's do 5 restarts.
    num_restarts = 5
    
    for r in range(num_restarts):
        # Initialize with random signs
        # Bias towards +1 or -1? Usually symmetric.
        current_signs = np.random.choice([-1.0, 1.0], size=N)
        
        # Run local search
        # Time budget per restart
        time_per_restart = (timeout - 5.0) / num_restarts
        local_start = time.time()
        
        improved = True
        iters = 0
        max_iters = 5000 # Safety cap
        
        while improved and iters < max_iters and (time.time() - local_start) < time_per_restart:
            current_signs, current_peak, improved = _vectorized_local_search(current_signs)
            iters += 1
            
        current_score = current_peak
        
        if current_score < best_score:
            best_score = current_score
            best_signs = current_signs.copy()
            
        # Check global time
        if time.time() - start_time > timeout - 10.0:
            break
            
    return best_signs
