import numpy as np

N = 400

def construct():
    # Initialize with random real vector
    np.random.seed(42)
    x = np.random.randn(N)
    
    # We want to maximize the denominator (sum x)^2, so let's start with positive bias
    x = np.abs(x)
    x = x / np.linalg.norm(x)
    
    M = 2 * N  # Zero-padding for linear autocorrelation
    
    best_x = x.copy()
    best_score = np.inf
    
    # ADMM-like parameters
    lambda_l1 = 0.05  # Sparsity penalty (increased to encourage step-like behavior)
    step_size = 0.5   # Learning rate for gradient step
    
    for iteration in range(500):
        # 1. Compute Autocorrelation
        x_pad = np.zeros(M)
        x_pad[:N] = x
        X = np.fft.fft(x_pad)
        P = np.abs(X)**2
        R = np.fft.ifft(P).real
        
        # Extract linear autocorrelation
        # R[0] is lag 0
        # R[1:N] are lags 1 to N-1
        # R[M-N+1:M] are lags -(N-1) to -1
        R_sidelobes = np.concatenate([R[1:N], R[M-N+1:M]])
        
        # Calculate current score (max abs sidelobe / sum^2)
        sum_x = np.sum(x)
        if sum_x == 0:
            sum_x = 1e-9
        score = np.max(np.abs(R_sidelobes)) / (sum_x**2)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
            
        # Find the lag with the maximum absolute sidelobe
        idx_max = np.argmax(np.abs(R_sidelobes))
        # Map index back to actual lag
        # R_sidelobes structure: [R[1], ..., R[N-1], R[M-N+1], ..., R[M-1]]
        # Length is 2*(N-1)
        if idx_max < N - 1:
            k_max = idx_max + 1
        else:
            k_max = M - (idx_max - (N - 1))
            # Negative lag
            k_max = -k_max
            
        # Gradient of R(k) with respect to x
        # R(k) = sum x_i x_{i+k}
        # grad = shift(x, k) + shift(x, -k)
        # Using circular shifts on the padded vector for efficiency
        # We need to be careful with linear vs circular.
        # Since we zero-padded, circular shift of the padded vector corresponds to linear shift + wrap-around garbage.
        # However, for the gradient of the *linear* autocorrelation, we can compute it directly or via FFT of the gradient kernel.
        
        # Efficient gradient computation:
        # The gradient of R(k) is equivalent to the cross-correlation of x with a delta function at k.
        # Let's construct the gradient vector g.
        g = np.zeros(N)
        if k_max > 0:
            # x_i contributes to R(k) via x_{i+k}
            # dR/dx_i = x_{i+k} (if i+k < N) + x_{i-k} (if i-k >= 0)
            # Term 1: x shifted left by k (x[i+k])
            g[:N-k_max] += x[k_max:]
            # Term 2: x shifted right by k (x[i-k])
            g[k_max:] += x[:N-k_max]
        elif k_max < 0:
            k_abs = -k_max
            # Same logic, symmetry of autocorrelation
            g[:N-k_abs] += x[k_abs:]
            g[k_abs:] += x[:N-k_abs]
            
        # Sign of the peak to determine descent direction
        peak_val = R_sidelobes[idx_max]
        sign_peak = np.sign(peak_val)
        
        # Gradient step
        x_temp = x - step_size * sign_peak * g
        
        # Proximal Operator for L1 norm (Soft Thresholding)
        threshold = lambda_l1 / step_size
        x_new = np.sign(x_temp) * np.maximum(np.abs(x_temp) - threshold, 0)
        
        # Enforce non-negativity to maximize Sum (Denominator)
        # "Signed step functions" might allow negatives, but for this score,
        # unipolar is superior for the denominator.
        # We clip to 0 to maintain large sum.
        x_new = np.maximum(x_new, 0)
        
        # Normalize to unit norm to keep scale consistent
        norm_x = np.linalg.norm(x_new)
        if norm_x > 1e-6:
            x_new = x_new / norm_x
        else:
            # Fallback if all zeros
            x_new = np.ones(N) / np.sqrt(N)
            
        x = x_new
        
    return best_x
