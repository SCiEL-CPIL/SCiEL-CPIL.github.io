import numpy as np
import time

N = 600

def _autoconv(h):
    """Compute discrete autoconvolution of h (length N).
    Returns array of length 2N-1 where c[k] = sum_i h[i]*h[i+k] for k in [-(N-1), N-1].
    We return c for lags 0..N-1 (symmetric)."""
    # Full linear convolution with itself
    c = np.convolve(h, h, mode='full')
    return c  # length 2N-1, c[k] = sum_i h[i]*h[i+k-N+1]... let me be careful

def _autoconv_max_norm(h):
    """Compute the normalized maximum of the autoconvolution.
    Normalization: max(c) / (sum(h))^2 * N, or similar.
    For uniform h=1, sum=N, c[center]=N, so normalized = N/N^2 * N = 1.
    Let's use: max(c) * N / (sum(h))^2
    For uniform: max(c)=N, sum=N, so N*N/N^2 = 1. Good.
    """
    c = np.convolve(h, h, mode='full')
    max_c = np.max(c)
    s = np.sum(h)
    if s < 1e-12:
        return 1e10
    return max_c * N / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    # Initial guess: uniform
    h = np.ones(N)
    best_h = h.copy()
    best_val = _autoconv_max_norm(h)
    
    # Iterative peak-reduction (Remez-like / water-filling)
    # At each iteration:
    # 1. Compute autoconvolution
    # 2. Find peak lags
    # 3. Compute sensitivity (gradient) at peak lags
    # 4. Adjust h: decrease where sensitive, increase where not, maintaining nonnegativity
    
    alpha = 0.1  # step size
    for it in range(2000):
        if time.time() > deadline - 5.0:
            break
        
        # Normalize h to have sum = N (so uniform is reference)
        s = np.sum(h)
        if s < 1e-12:
            h = np.ones(N)
            s = N
        h = h * (N / s)
        
        # Compute autoconvolution
        c = np.convolve(h, h, mode='full')
        # c has length 2N-1. c[k] for k=0..2N-2 corresponds to lag k-(N-1)
        # The maximum is what we care about
        max_c = np.max(c)
        current_val = max_c * N / (s * s)
        
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
        
        # Find lags where c is close to max
        threshold = max_c * 0.95
        peak_indices = np.where(c >= threshold)[0]
        
        # Compute gradient of c[k] w.r.t. h[i] for peak lags
        # c[k] = sum_i h[i] * h[i + k - (N-1)]  ... let me index properly
        # For np.convolve(h, h, 'full'), result[m] = sum_i h[i] * h[m-i]
        # So result[m] = sum_{i=max(0,m-N+1)}^{min(N-1,m)} h[i]*h[m-i]
        # Gradient w.r.t. h[j] is sum over i where both i and m-i are valid:
        # d(result[m])/dh[j] = h[m-j] if 0<=m-j<N, else 0 (plus symmetric, but since it's autoconv, it's h[m-j])
        
        # For each peak lag m, the contribution to h[j] is h[m-j] if valid
        # We want to compute for each h[j], the total "sensitivity" = sum over peak m of h[m-j] (if valid)
        
        # Vectorized: for each j, sensitivity[j] = sum over peak m where 0<=m-j<N of h[m-j]
        # Let k = m-j, so k ranges over valid values, and m = j+k
        # sensitivity[j] = sum over k where 0<=j+k<2N-1 and (j+k) is a peak index of h[k]
        
        # Build a mask for peak indices
        peak_mask = np.zeros(2*N-1, dtype=bool)
        peak_mask[peak_indices] = True
        
        # For each j, sensitivity[j] = sum_{k: peak_mask[j+k] and 0<=j+k<2N-1} h[k]
        # This is like a cross-correlation
        # sensitivity = np.correlate(h, peak_mask.astype(float), mode='same')... not quite
        
        # Let's do it directly: for each j, we need sum of h[k] for k such that j+k is in peak_indices
        # Equivalent: sensitivity[j] = sum_k h[k] * peak_mask[j+k]
        # This is a convolution-like operation
        
        # Use FFT for speed
        from scipy import fft
        H = fft.fft(h, n=4*N)
        PM = fft.fft(peak_mask.astype(float), n=4*N)
        sens = np.real(fft.ifft(H * PM))
        # sens[j] = sum_k h[k] * peak_mask[j+k] ... need to check indexing
        # Actually np.convolve convention: (a*b)[n] = sum_k a[k]b[n-k]
        # We want sum_k h[k] peak_mask[j+k] = sum_k h[k] peak_mask[(j+k)]
        # Let m=j+k, then sum_m h[m-j] peak_mask[m] = (h_reversed * peak_mask)[j]... 
        # This is getting complicated. Let's just do it with a loop over peaks if few, or use a different approach.
        
        # Simpler: for each peak index m, add h[m-j] to sensitivity[j] for valid j
        # If there are P peaks, this is O(P*N) which for P~100, N=600 is 60000 ops per iteration - fine
        
        sensitivity = np.zeros(N)
        for m in peak_indices:
            # c[m] = sum_i h[i]*h[m-i] where 0<=i<N and 0<=m-i<N
            # d(c[m])/dh[j] = h[m-j] if 0<=m-j<N
            k_min = max(0, m - (N-1))
            k_max = min(N-1, m)
            for j in range(k_min, k_max+1):
                sensitivity[j] += h[m-j]
        
        # Now adjust h: we want to reduce the peak.
        # Decrease h where sensitivity is high, increase where low.
        # Use water-filling: new_h = h - alpha * (sensitivity - mean(sensitivity))
        # But ensure nonnegativity
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        # Normalize sensitivity to have zero mean, unit std
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Step: h_new = h - alpha * sens_norm * h (proportional to current value)
        # Or: h_new = h * (1 - alpha * sens_norm)
        h_new = h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        # Check if this improved
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        c_new = np.convolve(h_new, h_new, mode='full')
        val_new = np.max(c_new) * N / (s_new * s_new)
        
        if val_new < current_val:
            h = h_new
        else:
            # Reduce step size
            alpha *= 0.9
            if alpha < 1e-6:
                break
    
    # Final: return the best found
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0.0)
    return [float(x) for x in best_h]
