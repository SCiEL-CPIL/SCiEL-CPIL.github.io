<<<<<<< SEARCH
import numpy as np
import time

N = 600

def compute_ratio(heights):
    """
    Computes the normalized maximum of the autoconvolution.
    Ratio = max(autoconv) / sum(heights^2)
    """
    if np.max(heights) == 0:
        return 1.0
    autoconv = np.correlate(heights, heights, mode='full')
    max_val = np.max(autoconv)
    norm_sq = np.sum(heights**2)
    return max_val / norm_sq

def construct():
    # The theoretical minimum for non-negative functions is 1.0.
    # This is achieved by any non-negative function because the 
    # autoconvolution of a non-negative function is maximized at lag 0,
    # which is equal to the sum of squares.
    # Thus ratio = max(autoconv) / sum(heights^2) = 1.
    
    # We return a constant function to be safe and deterministic.
    return [1.0] * N
=======
import numpy as np
import time

N = 600
DEADLINE = 140.0  # Safety margin before 150s kill

def compute_ratio(heights):
    """
    Computes the normalized maximum of the autoconvolution.
    Ratio = max(autoconv) / sum(heights^2)
    """
    if np.max(heights) == 0:
        return 1.0
    autoconv = np.correlate(heights, heights, mode='full')
    max_val = np.max(autoconv)
    norm_sq = np.sum(heights**2)
    return max_val / norm_sq

def smooth_max_surrogate(x, temperature=0.1):
    """
    Smooth approximation of max(x) using log-sum-exp.
    """
    x = np.asarray(x)
    max_x = np.max(x)
    if max_x == 0:
        return 0.0
    shifted = (x - max_x) / temperature
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    log_sum_exp = max_x + temperature * np.log(np.sum(np.exp(shifted)))
    return log_sum_exp

def objective(heights, temperature=0.1):
    """
    Objective: normalized smooth max of autoconvolution.
    """
    heights = np.asarray(heights)
    autoconv = np.correlate(heights, heights, mode='full')
    norm_sq = np.sum(heights**2)
    if norm_sq == 0:
        return 1.0
    smooth_max = smooth_max_surrogate(autoconv, temperature)
    return smooth_max / norm_sq

def objective_grad(heights, temperature=0.1):
    """
    Gradient of the normalized smooth max objective with respect to heights.
    Uses the chain rule: d/dh [max(autoconv) / ||h||^2]
    """
    heights = np.asarray(heights)
    autoconv = np.correlate(heights, heights, mode='full')
    norm_sq = np.sum(heights**2)
    if norm_sq == 0:
        return np.zeros_like(heights)
    
    # Gradient of smooth max w.r.t. autoconv
    max_x = np.max(autoconv)
    if max_x == 0:
        return np.zeros_like(heights)
    shifted = (autoconv - max_x) / temperature
    shifted = np.clip(shifted, -50, 50)
    weights = np.exp(shifted)
    weights = weights / np.sum(weights)
    
    # Gradient of autoconv w.r.t. heights
    # autoconv[i] = sum_j heights[j] * heights[j+i-N] (for appropriate indices)
    # d(autoconv[k])/d(heights[i]) = heights[i+k-N] + heights[i-k+N] (with boundary handling)
    # More precisely, autoconv = correlate(h, h, 'full')
    # autoconv[k] = sum_j h[j] * h[j-(k-N)] where indices are valid
    
    # The gradient of the quadratic form h^T Q h w.r.t. h is 2*Q*h
    # where Q is the matrix such that autoconv = h^T Q h
    # For autoconvolution, this is equivalent to the convolution matrix
    
    # d(autoconv)/d(h) can be computed via:
    # If autoconv[k] = sum_i h[i] * h[i + (k - N)] (with bounds)
    # Then d(autoconv[k])/d(h[j]) = h[j + (k-N)] + h[j - (k-N)] (if indices valid)
    
    # Vectorized: grad_autoconv[i] = sum_k weights[k] * (h[i + (k-N)] + h[i - (k-N)])
    # This is equivalent to weights convolved with h, plus reversed weights convolved with h
    
    # Let's compute this efficiently
    # autoconv has length 2N-1, indexed from 0 to 2N-2
    # autoconv[k] = sum_{j} h[j] * h[j + k - N + 1] (for valid j)
    
    # The gradient contribution from each lag k to h[i] is:
    # h[i + k - N + 1] + h[i - k + N - 1] (if indices in [0, N-1])
    
    # This is equivalent to:
    # grad[i] = sum_k weights[k] * (h[i + (k-N+1)] + h[i - (k-N+1)])
    
    # Let lags = np.arange(-N+1, N)
    # For each lag l = k - N + 1, the contribution is h[i+l] + h[i-l]
    
    # We can compute this as:
    # grad = convolve(weights, h) + convolve(weights[::-1], h)
    # But we need to be careful with the indexing
    
    # Actually, let's think more carefully:
    # autoconv[k] = sum_j h[j] * h[j + k - (N-1)] for k in [0, 2N-2]
    # Let l = k - (N-1), so l ranges from -(N-1) to N-1
    # autoconv[l] = sum_j h[j] * h[j+l] (with h out of bounds = 0)
    
    # d(autoconv[l])/d(h[i]) = h[i+l] + h[i-l] (if indices valid)
    
    # So grad_obj[i] = (1/norm_sq) * sum_l weights[l] * (h[i+l] + h[i-l]) - (2*smooth_max/norm_sq^2) * h[i]
    
    # The first term is: convolve(weights, h) + convolve(weights_flipped, h)
    # where weights is indexed by l from -(N-1) to N-1
    
    # Let's construct the weight vector properly
    # weights corresponds to autoconv indices 0 to 2N-2
    # lag l = k - (N-1) for k in 0..2N-2
    
    # Create a vector w of length 2N-1 where w[k] = weights[k]
    # Then the first sum is: sum_k w[k] * h[i + k - (N-1)]
    # This is a convolution of w with h, but we need to align properly
    
    # Let's use a direct approach with FFT for efficiency
    # Or just compute it as a matrix-vector product
    
    # For N=600, we can afford O(N^2) in some cases, but let's try to be efficient
    
    # The term sum_l weights[l] * h[i+l] is a cross-correlation
    # Let's define w_full as an array of length 2N-1
    # w_full[k] = weights[k] for k in 0..2N-2
    
    # Then sum_k w_full[k] * h[i + k - (N-1)] 
    # = sum_j w_full[j - i + (N-1)] * h[j] (changing variable)
    # This is (w_full_shifted * h) where w_full_shifted is w_full shifted
    
    # Actually, let's just compute it directly using the fact that:
    # grad_autoconv[i] = sum_{k=0}^{2N-2} weights[k] * (h[i + k - N + 1] + h[i - k + N - 1])
    
    # Let's compute this as two convolutions
    # Term 1: sum_k weights[k] * h[i + k - N + 1]
    # Let j = i + k - N + 1, so k = j - i + N - 1
    # = sum_j weights[j - i + N - 1] * h[j]
    # This is a convolution of the sequence weights with h, but with a shift
    
    # Let's create an array w of length 2N-1
    # w[k] = weights[k]
    # Then term1[i] = sum_k w[k] * h[i + k - (N-1)]
    
    # This is equivalent to: (w_conv * h)[i] where w_conv is w reversed and shifted
    
    # Let's use scipy.signal or just compute via FFT
    
    from scipy.signal import fftconvolve
    
    # w has length 2N-1, h has length N
    # term1[i] = sum_{k} w[k] * h[i + k - (N-1)]
    # Let m = i - (N-1), then term1[i] = sum_k w[k] * h[m + k]
    # This is (w * h)[m] = (w * h)[i - (N-1)]
    # where * denotes convolution
    
    w = weights  # length 2N-1
    # convolve w with h
    conv_wh = fftconvolve(w, h, mode='full')
    # conv_wh has length 3N-2
    # conv_wh[m] = sum_k w[k] * h[m - k] ... wait, need to check convention
    
    # Actually, fftconvolve(w, h)[n] = sum_k w[k] * h[n-k]
    # We want sum_k w[k] * h[i + k - (N-1)]
    # Let j = i + k - (N-1), so k = j - i + (N-1)
    # = sum_j w[j - i + (N-1)] * h[j]
    # This is (w_rev * h)[i] where w_rev[j] = w[(N-1) - j] ... 
    
    # Let me just do it directly for clarity, since N=600 is small enough
    # for a vectorized approach
    
    # Create index matrix
    # For each i in 0..N-1 and k in 0..2N-2:
    # idx1 = i + k - (N-1)
    # if 0 <= idx1 < N: contribution += weights[k] * h[idx1]
    
    # Vectorized:
    i_idx = np.arange(N)[:, None]  # N x 1
    k_idx = np.arange(2*N-1)[None, :]  # 1 x (2N-1)
    idx1 = i_idx + k_idx - (N-1)  # N x (2N-1)
    mask1 = (idx1 >= 0) & (idx1 < N)
    
    # term1[i] = sum_k weights[k] * h[idx1[i,k]] * mask1[i,k]
    # h_idx = np.take(h, np.clip(idx1, 0, N-1))
    h_idx1 = np.where(mask1, h[np.clip(idx1, 0, N-1)], 0.0)
    term1 = np.sum(weights[None, :] * h_idx1, axis=1)
    
    # Term 2: sum_k weights[k] * h[i - k + (N-1)]
    idx2 = i_idx - k_idx + (N-1)
    mask2 = (idx2 >= 0) & (idx2 < N)
    h_idx2 = np.where(mask2, h[np.clip(idx2, 0, N-1)], 0.0)
    term2 = np.sum(weights[None, :] * h_idx2, axis=1)
    
    grad_autoconv = term1 + term2
    
    # Now the full gradient of (smooth_max / norm_sq)
    # = grad_autoconv / norm_sq - smooth_max * 2*h / norm_sq^2
    grad = grad_autoconv / norm_sq - (smooth_max * 2 * h) / (norm_sq ** 2)
    
    return grad

def frank_wolfe_step(h, grad, temperature=0.1):
    """
    Frank-Wolfe step: minimize linear approximation over the simplex.
    """
    h = np.asarray(h)
    # The simplex constraint: h >= 0, sum(h) = 1
    # Linear minimization: min v^T s subject to s >= 0, sum(s) = 1
    # Optimal: s = e_{argmin v}
    min_idx = np.argmin(grad)
    s = np.zeros(N)
    s[min_idx] = 1.0
    return s

def construct():
    start_time = time.time()
    
    # Initialize with uniform distribution
    h = np.ones(N) / N
    best_h = h.copy()
    best_score = compute_ratio(h)
    
    # Adaptive temperature schedule
    temperature = 0.05
    
    # Frank-Wolfe iterations
    max_iter = 500
    step_size = 1.0  # For FW, we use the standard step size t_k = 2/(k+2)
    
    for it in range(max_iter):
        if time.time() - start_time > DEADLINE:
            break
        
        # Compute objective and gradient
        obj_val = objective(h, temperature)
        grad = objective_grad(h, temperature)
        
        # Check actual ratio
        actual_ratio = compute_ratio(h)
        if actual_ratio < best_score:
            best_score = actual_ratio
            best_h = h.copy()
        
        # Frank-Wolfe step
        s = frank_wolfe_step(h, grad, temperature)
        
        # Step size
        t = 2.0 / (it + 2)
        
        # Update
        h_new = (1 - t) * h + t * s
        
        # Project to simplex (ensure non-negative and sum to 1)
        h_new = np.maximum(h_new, 0)
        if np.sum(h_new) > 0:
            h_new = h_new / np.sum(h_new)
        else:
            h_new = np.ones(N) / N
        
        h = h_new
        
        # Reduce temperature over time
        if it % 50 == 0 and it > 0:
            temperature *= 0.8
    
    # Final polish: return the best found
    return list(best_h)
>>>>>>> REPLACE
