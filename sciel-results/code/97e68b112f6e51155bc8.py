"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initialize with uniform heights
    h = np.ones(N)
    
    # Normalize: sum of heights = 1 (for normalization)
    h = h / np.sum(h)
    
    def autoconv(h):
        """Compute discrete autoconvolution coefficients."""
        # Full autoconvolution has 2*N-1 lags
        # Use FFT for efficiency
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        ac = np.fft.irfft(H * np.conj(H), n=n_fft)
        # The autoconvolution is centered; extract the relevant part
        # ac[0] is the sum of squares (lag 0)
        # ac[1:N] are positive lags, ac[n_fft-N+1:] are negative lags (same as positive)
        return ac
    
    def normalized_max_autoconv(h):
        """Compute normalized maximum of autoconvolution."""
        ac = autoconv(h)
        # The integral of f is sum(h) * interval_width, but since we normalize h to sum=1,
        # the normalization should account for the interval width
        # Let's use max(ac) / (sum(h))^2 as a proxy, or just max(ac) since sum(h)=1
        # Actually for step functions, the autoconvolution value at lag 0 is sum(h^2)
        # and the "normalized" version might be max(ac) / sum(h)^2
        s = np.sum(h)
        if s == 0:
            return np.inf
        return np.max(ac) / (s * s)
    
    best_h = h.copy()
    best_score = normalized_max_autoconv(h)
    
    # Gradient descent with log-sum-exp smooth max
    def smooth_max_autoconv(h, beta=100.0):
        """Smooth approximation of max(autoconv) using log-sum-exp."""
        ac = autoconv(h)
        # Use log-sum-exp for numerical stability
        max_ac = np.max(ac)
        log_sum_exp = max_ac + np.log(np.sum(np.exp(ac - max_ac)))
        s = np.sum(h)
        if s == 0:
            return np.inf
        return log_sum_exp / (s * s)
    
    # Coordinate descent / gradient steps
    lr = 0.01
    for iteration in range(5000):
        if time.time() > deadline - 10.0:
            break
        
        # Compute gradient of smooth max autoconv w.r.t. h
        ac = autoconv(h)
        max_ac = np.max(ac)
        # Softmax weights for the smooth max
        exp_ac = np.exp(ac - max_ac)
        softmax_weights = exp_ac / np.sum(exp_ac)
        
        # Gradient of autoconv w.r.t. h
        # ac[k] = sum_i h[i] * h[i+k] (approximately)
        # d(ac[k])/d(h[j]) = h[j+k] + h[j-k] (with appropriate bounds)
        # This is the convolution of h with itself, so gradient is 2*convolve(h, h) appropriately
        # Actually, the gradient of the autoconvolution at each lag w.r.t. h is:
        # For the full autoconvolution computed via FFT, the gradient is the convolution of h with h
        # but we need to be careful about the exact relationship
        
        # Simpler: use the fact that d/dh_j [sum_k w_k * ac[k]] where ac = ifft(fft(h)^2)
        # The gradient can be computed as: grad = 2 * ifft(fft(h) * fft(h) * fft(weights))
        # but this is tricky. Let me use a different approach.
        
        # Alternative: use subgradient of the max
        # The subgradient of max(ac) w.r.t. h is the column of the Jacobian corresponding to the max lag
        # For simplicity, let's use coordinate descent instead
        
        # Coordinate descent: try adjusting each h[i]
        improved = False
        for i in range(0, N, 2):  # Step by 2 for speed, or all if time permits
            if time.time() > deadline - 15.0:
                break
            
            # Try small perturbations
            for delta in [0.001, -0.001]:
                h_new = h.copy()
                h_new[i] += delta
                if h_new[i] < 0:
                    continue
                score = normalized_max_autoconv(h_new)
                if score < best_score - 1e-10:
                    h = h_new
                    best_score = score
                    best_h = h.copy()
                    improved = True
                    break
        
        if not improved and iteration > 100:
            # Try smaller steps
            for i in range(0, N, 4):
                if time.time() > deadline - 15.0:
                    break
                for delta in [0.0001, -0.0001]:
                    h_new = h.copy()
                    h_new[i] += delta
                    if h_new[i] < 0:
                        continue
                    score = normalized_max_autoconv(h_new)
                    if score < best_score - 1e-12:
                        h = h_new
                        best_score = score
                        best_h = h.copy()
                        break
    
    # Return the best heights found
    return list(best_h)
# EVOLVE-BLOCK-END
