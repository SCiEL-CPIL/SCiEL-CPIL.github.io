"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: ACTIVE-SET GRADIENT DESCENT
- The objective J(f) = max_k (f*f)_k / (sum f)^2 is non-smooth.
- We track the "active set" of lags K where the convolution is close to the maximum.
- For a fixed K, J_K(f) is smooth. We compute the exact gradient of J_K(f).
- We perform gradient descent with line search on f, keeping f >= 0.
- We periodically update the active set K.
- This is structurally different from random parametric search; it directly
  optimizes the 600-dimensional vector using first-order information.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 140  # Safety margin for 150s limit

def compute_score_exact(f):
    """Compute the normalized maximum of the autoconvolution exactly."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def compute_conv(f):
    """Compute full autoconvolution."""
    return np.convolve(f, f, mode='full')

def gradient_active_set(f, active_lags):
    """
    Compute the gradient of J_K(f) = max_{k in K} (f*f)_k / (sum f)^2
    assuming the max is achieved at lags in active_lags.
    Note: Since max is not differentiable, we use the gradient of the average
    of the active terms or the gradient of the single max term. 
    Using the average of active terms is often more stable for descent.
    However, for minimizing the max, we should focus on the steepest increasing
    direction. Let's use the gradient of the mean of the top few active lags.
    """
    s = np.sum(f)
    if s < 1e-12:
        return np.zeros(N)
    
    conv = compute_conv(f)
    # Normalize conv to get the term (f*f)_k / s^2
    # J = max_k C_k / s^2
    # d/d f_j (C_k / s^2) = (1/s^2) * dC_k/df_j - (C_k/s^3) * 2 * (1/s) * ds/df_j
    # ds/df_j = 1
    # dC_k/df_j = sum_m (df_m/df_j * f_{k-m} + f_m * df_{k-m}/df_j)
    # Since f is the vector, C_k = sum_{m=0}^{N-1} f_m f_{k-m} (with bounds)
    # dC_k/df_j = f_{k-j} + f_{j-k} (if indices valid)
    
    # Let's compute the gradient of the objective for a specific lag k
    # grad_k = (1/s^2) * g_k - 2 * (C_k/s^2) * (1/s) * 1
    # where g_k is the vector with components g_k[j] = f_{k-j} + f_{j-k}
    
    # We want to minimize the max. A common technique is to minimize the
    # log-sum-exp or just the mean of the top active lags.
    # Let's use the mean of the active lags for the gradient direction.
    
    grad = np.zeros(N)
    count = 0
    for k in active_lags:
        # Compute g_k vector
        # g_k[j] = f[k-j] if 0<=k-j<N else 0 + f[j-k] if 0<=j-k<N else 0
        # Note: convolution is defined as sum_{m} f[m] f[k-m].
        # Indices m from 0 to N-1, k-m from 0 to N-1.
        
        # Vectorized computation of g_k
        # f[k-j] is non-zero if 0 <= k-j < N => k-N < j <= k
        # f[j-k] is non-zero if 0 <= j-k < N => k < j <= k+N
        
        # Instead of looping k, let's just compute the gradient for the single max lag
        # or the top few. For simplicity and speed, let's use the top 5 lags.
        pass
        
    # More efficient: compute gradient for the specific lags in active_lags
    # We can do this by iterating over active_lags and adding contributions.
    # But active_lags might be large. Let's restrict to top 10 lags.
    
    # Actually, let's just compute the gradient of the SINGLE maximum lag.
    # If there are ties, it's still a valid subgradient direction for descent.
    
    max_idx = np.argmax(conv)
    k_max = max_idx
    C_k = conv[k_max]
    
    # Gradient of C_k / s^2
    # d/d f_j (C_k s^{-2}) = s^{-2} g_k[j] - 2 C_k s^{-3}
    
    # Compute g_k vector
    # g_k[j] = f[k-j] + f[j-k] (with 0 for out of bounds)
    g_k = np.zeros(N)
    
    # Part 1: f[k-j]
    # j ranges such that 0 <= k-j < N
    # j <= k and j > k-N
    j_start = max(0, k - N + 1)
    j_end = min(N, k + 1)
    if j_start < j_end:
        # f[k-j] for j in [j_start, j_end)
        # indices of f are k - j
        idx_f = k - np.arange(j_start, j_end)
        g_k[j_start:j_end] += f[idx_f]
        
    # Part 2: f[j-k]
    # j ranges such that 0 <= j-k < N
    # j >= k and j < k+N
    j_start2 = max(0, k)
    j_end2 = min(N, k + N)
    if j_start2 < j_end2:
        # f[j-k] for j in [j_start2, j_end2)
        # indices of f are j - k
        idx_f2 = np.arange(j_start2, j_end2) - k
        g_k[j_start2:j_end2] += f[idx_f2]
        
    # Now combine
    # grad_j = (1/s^2) * g_k[j] - 2 * (C_k/s^2) * (1/s)
    # Note: The second term is constant for all j, so it doesn't affect the direction
    # of steepest descent in the affine space, but we must respect constraints.
    # Actually, it does affect the line search.
    
    base = (1.0 / (s * s))
    const_term = 2.0 * C_k / (s * s * s)
    
    grad = base * g_k - const_term
    
    return grad

def construct():
    start_time = time.time()
    
    # Initialize with a smooth random function
    rng = np.random.RandomState(42)
    # Start with a Gaussian-like bump to avoid zero
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    f = np.exp(-10 * x**2)
    f = np.maximum(f, 0.0)
    
    # Normalize to sum=1 for stability in early steps, though scale invariant
    f = f / np.sum(f) * N
    
    best_f = f.copy()
    best_score = compute_score_exact(f)
    
    # Gradient Descent with Active Set
    # We will iterate:
    # 1. Compute conv, find max lag k*
    # 2. Compute gradient of J at k*
    # 3. Line search for step size alpha
    # 4. Update f = max(0, f - alpha * grad)
    # 5. Repeat
    
    alpha = 1e-4
    min_alpha = 1e-10
    
    # Precompute FFT for faster conv if needed, but N=600 is small enough for direct
    # np.convolve is C-optimized and fast.
    
    for i in range(10000): # Max iterations
        if time.time() - start_time > TIME_LIMIT:
            break
            
        s = np.sum(f)
        if s < 1e-12:
            break
            
        conv = compute_conv(f)
        max_c = np.max(conv)
        current_score = max_c / (s * s)
        
        if current_score < best_score:
            best_score = current_score
            best_f = f.copy()
            
        # Find active lags (top few) to make gradient more robust
        # If we just use the single max, we might oscillate.
        # Let's use the top 5 lags for the gradient direction.
        # This effectively minimizes the average of the top peaks.
        
        # Sort conv indices by value
        # For efficiency, just find the max index. 
        # Using top-K is better. Let's use top 10.
        top_k = 10
        # np.argsort is O(N log N) which is fine for N=1200
        top_indices = np.argsort(conv)[-top_k:]
        
        # Compute gradient as average of gradients for these lags
        grad = np.zeros(N)
        for k in top_indices:
            # Compute g_k
            g_k = np.zeros(N)
            
            # f[k-j]
            j_start = max(0, k - N + 1)
            j_end = min(N, k + 1)
            if j_start < j_end:
                idx_f = k - np.arange(j_start, j_end)
                g_k[j_start:j_end] += f[idx_f]
                
            # f[j-k]
            j_start2 = max(0, k)
            j_end2 = min(N, k + N)
            if j_start2 < j_end2:
                idx_f2 = np.arange(j_start2, j_end2) - k
                g_k[j_start2:j_end2] += f[idx_f2]
                
            # Gradient of C_k / s^2
            # (1/s^2) g_k - 2 C_k / s^3
            # We accumulate the g_k part and the constant part separately?
            # No, just add the full gradient vector.
            
            base = 1.0 / (s * s)
            const_term = 2.0 * conv[k] / (s * s * s)
            
            grad += base * g_k - const_term
            
        grad /= top_k # Average
        
        # Line Search
        # We want to decrease current_score.
        # f_new = max(0, f - alpha * grad)
        
        found_alpha = False
        current_alpha = alpha
        
        for ls_iter in range(20):
            f_trial = f - current_alpha * grad
            f_trial = np.maximum(f_trial, 0.0)
            
            s_trial = np.sum(f_trial)
            if s_trial < 1e-12:
                current_alpha *= 0.5
                continue
                
            conv_trial = compute_conv(f_trial)
            score_trial = np.max(conv_trial) / (s_trial * s_trial)
            
            if score_trial < current_score:
                found_alpha = True
                f = f_trial
                # Increase alpha slightly if successful
                alpha = min(current_alpha * 1.1, 1e-2)
                break
            else:
                current_alpha *= 0.5
                if current_alpha < min_alpha:
                    break
                    
        if not found_alpha:
            # If line search fails, try a smaller alpha or break
            if current_alpha < min_alpha:
                # Try a random perturbation to escape local minimum?
                # Or just break.
                # Let's try a small random step
                f = f + rng.normal(0, 1e-5, N)
                f = np.maximum(f, 0.0)
            else:
                break
                
    # Final exact check
    final_score = compute_score_exact(best_f)
    if final_score < best_score:
        best_score = final_score
        
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    start = time.time()
    f = construct()
    elapsed = time.time() - start
    score = compute_score_exact(f)
    print(f"Score: {score}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
    print(f"Time: {elapsed:.2f}s")
