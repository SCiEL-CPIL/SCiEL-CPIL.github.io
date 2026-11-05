"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

TIME_LIMIT = 145  # safety margin before 150s kill


def _compute_convolution(h):
    """
    Compute the linear autoconvolution of h via FFT.
    Returns the convolution array of length 2*n-1.
    """
    n = len(h)
    h_pad = np.zeros(2 * n - 1)
    h_pad[:n] = h
    
    # Use next fast FFT size
    fft_size = 1
    while fft_size < 2 * (2 * n - 1):
        fft_size *= 2
    
    f_fft = np.fft.fft(h_pad, n=fft_size)
    conv = np.fft.ifft(f_fft * np.conj(f_fft)).real[:2 * n - 1]
    
    return conv


def _compute_c3_bound(heights):
    """
    Compute the upper bound on C3 from the autoconvolution of the signed step function.
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    dx = 0.5 / n  # total length 0.5, n intervals
    
    conv = _compute_convolution(h)
    max_abs = np.max(np.abs(conv))
    c3_bound = dx * max_abs
    
    return c3_bound


def _generate_fbm(n, H, seed=42):
    """
    Generate a deterministic Fractional Brownian Motion (fBm) sample of length n
    with Hurst exponent H, and return the sign of the samples.
    Uses Cholesky decomposition of the covariance matrix.
    """
    rng = np.random.RandomState(seed)
    
    # Covariance matrix for fBm:
    # Cov(X_i, X_j) = 0.5 * (|i|^(2H) + |j|^(2H) - |i-j|^(2H))
    # We can generate this efficiently.
    idx = np.arange(n)
    # Compute pairwise differences
    # To save memory, we can compute the covariance matrix directly for N=400 (400x400 is small)
    diff = np.abs(idx[:, None] - idx[None, :])
    cov = 0.5 * (np.power(idx[:, None], 2*H) + np.power(idx[None, :], 2*H) - np.power(diff, 2*H))
    
    # Cholesky decomposition
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # If not positive definite (shouldn't happen for valid H), add small jitter
        cov += 1e-10 * np.eye(n)
        L = np.linalg.cholesky(cov)
    
    # Generate standard normal variables
    z = rng.randn(n)
    # fBm samples
    fbm = L @ z
    
    # Convert to signed step function
    h = np.sign(fbm)
    h[h == 0] = 1.0  # Avoid zeros
    
    return h


def _fast_local_search(heights, max_iter=10000, time_budget=60, start_time=0):
    """
    Fast local search using incremental convolution updates.
    When flipping bit i, the convolution changes by -2*h[i] * (h convolved with delta_i).
    This allows O(N) update instead of O(N log N) FFT per evaluation.
    """
    h = np.array(heights, dtype=np.float64)
    n = len(h)
    conv_len = 2 * n - 1
    
    # Precompute convolution
    conv = _compute_convolution(h)
    current_max = np.max(np.abs(conv))
    current_bound = (0.5 / n) * current_max
    
    if start_time == 0:
        start_time = time.time()
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        iters += 1
        best_gain = 0
        best_idx = -1
        best_new_max = current_max
        
        # Precompute the "influence" of each position on the convolution
        # Flipping position i changes conv by -2*h[i] * h shifted
        # We can compute this incrementally
        
        for i in range(n):
            # Compute the effect of flipping position i
            # New conv = conv - 2*h[i] * (h convolved with delta at i)
            # h convolved with delta_i is just h shifted by i positions
            # For linear convolution of length 2n-1:
            # (h * delta_i)[k] = h[k-i] if 0 <= k-i < n, else 0
            
            # The change in conv when flipping h[i]:
            # delta_conv[k] = -2 * h[i] * h[k-i] for valid k-i
            # We need to find the new max |conv + delta_conv|
            
            # Efficient approach: compute the affected range
            # h[k-i] is nonzero for k in [i, i+n-1]
            start_k = i
            end_k = min(i + n, conv_len)
            
            # Extract the affected portion
            affected = conv[start_k:end_k].copy()
            # The shift of h by i: h[k-i] for k in [i, i+n-1]
            # h[k-i] = h[0], h[1], ..., h[n-1]
            delta = -2.0 * h[i] * h[:end_k - start_k]
            
            new_affected = affected + delta
            
            # Find new max
            # The unaffected parts remain the same
            if start_k > 0:
                max_left = np.max(np.abs(conv[:start_k]))
            else:
                max_left = 0.0
            if end_k < conv_len:
                max_right = np.max(np.abs(conv[end_k:]))
            else:
                max_right = 0.0
            
            max_affected = np.max(np.abs(new_affected)) if len(new_affected) > 0 else 0.0
            new_max = max(max_left, max_affected, max_right)
            
            gain = current_max - new_max
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                best_new_max = new_max
                best_delta = delta
                best_start = start_k
                best_end = end_k
        
        if best_idx >= 0:
            # Apply the flip
            h[best_idx] = -h[best_idx]
            conv[best_start:best_end] += best_delta
            current_max = best_new_max
            current_bound = (0.5 / n) * current_max
            improved = True
    
    return h, current_bound


def construct():
    start_time = time.time()
    
    # Initialize with fBm-based sequences
    # Coarse grid search over Hurst exponent H
    best_h = np.ones(N)
    best_bound = float('inf')
    
    # H in [0.1, 0.9]
    # H=0.5 is standard Brownian motion (white noise increments)
    # H < 0.5 is anti-persistent
    # H > 0.5 is persistent
    H_values = np.linspace(0.1, 0.9, 17)
    
    for H in H_values:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        # Try a few different seeds for each H to get variety
        for seed in [42, 123, 999]:
            if time.time() - start_time > TIME_LIMIT - 30:
                break
            
            h_fbm = _generate_fbm(N, H, seed=seed)
            bound = _compute_c3_bound(h_fbm)
            
            if bound < best_bound:
                best_bound = bound
                best_h = h_fbm.copy()
    
    # Fast local search from the best fBm initialization
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 10:
        local_time = min(remaining - 5, 60)
        best_h, best_bound = _fast_local_search(best_h, max_iter=20000, time_budget=local_time, start_time=start_time)
    
    # If time permits, try a second round with a different best H or seed
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 15:
        # Pick a different seed for the best H found? Or just random restart?
        # Let's try a random restart to see if we can beat the fBm local optimum
        np.random.seed(2023)
        h_rand = np.random.choice([-1.0, 1.0], size=N)
        bound_rand = _compute_c3_bound(h_rand)
        
        if bound_rand < best_bound:
            local_time = min(remaining - 5, 30)
            h_rand, bound_rand = _fast_local_search(h_rand, max_iter=10000, time_budget=local_time, start_time=start_time)
            if bound_rand < best_bound:
                best_bound = bound_rand
                best_h = h_rand.copy()
    
    return best_h


# Test
if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"First 20: {h[:20]}")
    print(f"C3 bound: {_compute_c3_bound(h)}")
    print(f"Score: {1.0 / _compute_c3_bound(h)}")
# EVOLVE-BLOCK-END
