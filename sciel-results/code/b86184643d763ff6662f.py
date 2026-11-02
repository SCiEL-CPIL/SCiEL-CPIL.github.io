import time
import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft

N = 600

def _fft_convolve(h):
    """Compute autoconvolution via FFT, returns full-length convolution."""
    n = len(h)
    size = 2 * n - 1
    # Pad to next power of 2 for efficiency
    fsize = 1
    while fsize < size:
        fsize <<= 1
    H = fft(h, fsize)
    conv = np.real(ifft(H * H, fsize))
    return conv[:size]

def compute_score(h):
    s = np.sum(h)
    if s < 1e-12:
        return 0.0
    conv = _fft_convolve(h)
    max_conv = np.max(conv)
    return max_conv / (s * s)

def _smooth_max(conv, beta):
    """Smooth approximation of max using log-sum-exp."""
    m = np.max(conv)
    if m <= 0:
        return 0.0
    shifted = conv / m
    return m * np.log(np.sum(np.exp(beta * (shifted - 1.0)))) / beta

def _smooth_score(h, beta):
    s = np.sum(h)
    if s < 1e-12:
        return 1.0
    conv = _fft_convolve(h)
    sm = _smooth_max(conv, beta)
    return sm / (s * s)

def _smooth_grad(h, beta):
    """Gradient of smooth score with respect to h (before clipping to nonneg)."""
    s = np.sum(h)
    if s < 1e-12:
        return np.zeros_like(h)
    conv = _fft_convolve(h)
    m = np.max(conv)
    if m <= 0:
        return np.zeros_like(h)
    # d(smooth_max)/d(conv_k) = exp(beta*(conv_k/m - 1)) / sum(exp(beta*(conv_j/m - 1)))
    shifted = conv / m
    weights = np.exp(beta * (shifted - 1.0))
    wsum = np.sum(weights)
    dsm_dconv = weights / wsum
    
    # d(conv_k)/d(h_i): conv_k = sum_j h_j * h_{k-j} (with appropriate indexing)
    # For full convolution: conv[k] = sum_i h[i] * h[k-i] where h is zero-padded
    # d(conv[k])/d(h[i]) = h[k-i] + h[k-i] = 2*h[k-i] (symmetric)
    # Actually conv[k] = sum_j h[j]*h[k-j], so d/dh[i] = h[k-i] + (if i != k-i) h[k-i]
    # More carefully: d(conv[k])/d(h[i]) = h[k-i] + h[k-i] only if we consider both terms
    # conv[k] = sum_j h[j] h[k-j], derivative w.r.t. h[i]: h[k-i] + h[k-i] = 2*h[k-i]
    # But we need to handle the convolution indexing properly.
    
    # Use FFT-based approach for gradient
    # d(smooth_max)/d(h[i]) = sum_k dsm_dconv[k] * d(conv[k])/d(h[i])
    # d(conv[k])/d(h[i]) = 2 * h[k-i] (for the autoconvolution, considering both j=i and k-j=i)
    # Actually: conv[k] = sum_j h[j] * h[k-j]
    # d/dh[i] conv[k] = h[k-i] + h[k-i] = 2*h[k-i] (when both indices are valid)
    # This is equivalent to 2 * (h * h_reflected)[k] where h_reflected is h reversed
    
    # Let's compute the gradient using the fact that:
    # grad_h[i] = sum_k dsm_dconv[k] * (h[k-i] + h[k-i]) 
    #            = 2 * sum_k dsm_dconv[k] * h[k-i]
    # which is 2 * (dsm_dconv * h)[i] in convolution sense
    
    # Compute via FFT for efficiency
    fsize = 1
    size = 2 * N - 1
    while fsize < size:
        fsize <<= 1
    
    D = fft(dsm_dconv, fsize)
    H = fft(h, fsize)
    # The convolution of dsm_dconv with h (full mode)
    conv_grad = np.real(ifft(D * H, fsize))[:size]
    
    # d(smooth_max)/d(h[i]) = 2 * conv_grad[i] ... wait, need to be more careful
    # conv[k] = sum_j h[j] h[k-j]
    # d(conv[k])/d(h[i]) = h[k-i] (from j=i term) + h[k-i] (from k-j=i term) = 2*h[k-i]
    # So sum_k dsm_dconv[k] * 2*h[k-i] = 2 * (dsm_dconv conv h)[i]
    # where (a conv b)[i] = sum_k a[k] b[i-k] ... hmm, indexing
    # Actually (a * b)[n] = sum_k a[k] b[n-k]
    # We want sum_k dsm_dconv[k] * h[k-i] = sum_k dsm_dconv[k] * h[-(i-k)] 
    # Let m = -k, then = sum_m dsm_dconv[-m] * h[i-m] = (dsm_dconv_flipped * h)[i]
    
    # Simpler: just compute directly for now since N=600 is manageable
    # But let's use the FFT approach correctly
    # (a conv b)[n] = sum_k a[k] b[n-k]
    # We need sum_k dsm_dconv[k] h[k-i] = sum_k dsm_dconv[k] h[-(i-k)]
    # = sum_k dsm_dconv[-k] h[i-k] if we flip... 
    
    # Let me just do it directly with numpy for clarity, N=600 should be fast enough
    # Actually, let's use a different approach: 
    # grad[i] = 2 * sum_k dsm_dconv[k] * h[k-i]
    # This is 2 * (dsm_dconv_reversed conv h)[i] ... 
    
    # Let me just compute it directly - N=600, conv has 1199 elements
    # For each i, grad[i] = 2 * sum_k dsm_dconv[k] * h[k-i]
    # where h[k-i] is 0 if k-i is out of [0, N-1]
    
    # Vectorized approach using the convolution theorem properly:
    # sum_k dsm_dconv[k] * h[k-i] = (dsm_dconv * h_rev)[i] where h_rev[j] = h[N-1-j]
    # Actually this is getting complicated. Let me just do a simple loop with numpy.
    
    # Alternative: use the fact that the gradient of sum_k w[k] * (h*h)[k] w.r.t. h[i]
    # = 2 * sum_k w[k] * h[k-i]
    # We can compute this as 2 * (w_flipped conv h)[i] where w_flipped[k] = w[size-1-k]
    
    w_flipped = dsm_dconv[::-1]
    # (w_flipped conv h)[i] = sum_k w_flipped[k] h[i-k] = sum_k w[size-1-k] h[i-k]
    # Let j = size-1-k, k = size-1-j
    # = sum_j w[j] h[i-(size-1-j)] = sum_j w[j] h[i-size+1+j]
    # Hmm, this doesn't quite work out cleanly.
    
    # Let me just do it the straightforward way with broadcasting for N=600
    # This will be O(N^2) per gradient eval which is 360000 ops - fine for 600 elements
    indices = np.arange(N)
    k_idx = np.arange(len(dsm_dconv))
    # For each i in [0,N-1], grad[i] = 2 * sum_k dsm_dconv[k] * h[k-i] if 0<=k-i<N else 0
    # Create matrix: M[i,k] = h[k-i] if valid else 0
    # This is a Toeplitz-like matrix
    
    # Simpler: use np.correlate or just loop
    grad_sm = np.zeros(N)
    for i in range(N):
        # k ranges such that 0 <= k-i < N, i.e., i <= k < i+N
        k_min = max(0, i)
        k_max = min(len(dsm_dconv), i + N)
        if k_min < k_max:
            valid_k = np.arange(k_min, k_max)
            valid_h_idx = valid_k - i
            grad_sm[i] = 2.0 * np.dot(dsm_dconv[k_min:k_max], h[valid_h_idx])
    
    # Now smooth_score = smooth_max / s^2
    # d(smooth_score)/d(h[i]) = d(smooth_max)/d(h[i]) / s^2 - 2 * smooth_max * d(1/s^2)/d(h[i])
    # d(1/s^2)/d(h[i]) = -2/s^3
    # So = grad_sm[i] / s^2 + 2 * smooth_max * grad_sm[i] / s^3 ... wait
    # d(smooth_max / s^2)/d(h[i]) = (d(smooth_max)/d(h[i]) * s^2 - smooth_max * 2s * 1) / s^4
    # = d(smooth_max)/d(h[i]) / s^2 - 2 * smooth_max / s^3
    
    sm = m * np.log(wsum) / beta
    grad = grad_sm / (s * s) - 2.0 * sm / (s * s * s)
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with uniform as baseline
    best_h = np.ones(N, dtype=np.float64)
    best_score = compute_score(best_h)
    
    # FFT-based gradient optimization with smooth max surrogate
    # Use annealing on beta to go from smooth to sharp max
    
    def _clip_nonneg(h):
        return np.maximum(h, 0.0)
    
    # Multi-start with different initializations
    initializations = []
    
    # Uniform
    initializations.append(np.ones(N, dtype=np.float64))
    
    # Two-block symmetric
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5 * (0.5 / N)
    for c, w in [(0.20, 0.08), (0.18, 0.10), (0.15, 0.12), (0.22, 0.06), (0.10, 0.15)]:
        h_init = np.zeros(N)
        mask1 = (x >= c - w) & (x <= c + w)
        mask2 = (x >= -c - w) & (x <= -c + w)
        h_init[mask1] = 1.0
        h_init[mask2] = 1.0
        initializations.append(h_init)
    
    # Cosine window
    x_cos = np.linspace(0, 2*np.pi, N)
    h_cos = 0.5 * (1 + np.cos(x_cos))
    initializations.append(h_cos)
    
    # Ramp-like
    h_ramp = np.linspace(0.1, 1.0, N)
    initializations.append(h_ramp)
    
    # Beta annealing schedule
    betas = [20, 50, 100, 200, 400]
    
    for h_init in initializations:
        if time.time() > deadline - 20:
            break
        
        h_curr = h_init.copy()
        
        for beta in betas:
            if time.time() > deadline - 15:
                break
            
            # Gradient descent with adaptive step size
            step = 0.1
            best_local = _smooth_score(h_curr, beta)
            best_h_local = h_curr.copy()
            
            max_iter = 200
            for it in range(max_iter):
                if time.time() > deadline - 15:
                    break
                
                h_curr = _clip_nonneg(h_curr)
                grad = _smooth_grad(h_curr, beta)
                
                # Normalize gradient for stable step
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-15:
                    break
                
                # Take a step
                h_new = _clip_nonneg(h_curr - step * grad / gnorm * 0.1)
                
                score_new = _smooth_score(h_new, beta)
                
                if score_new < best_local - 1e-10:
                    best_local = score_new
                    best_h_local = h_new.copy()
                    h_curr = h_new
                    # Decrease step slightly
                    step *= 0.99
                else:
                    # Increase step or try different direction
                    step *= 0.7
                    if step < 1e-6:
                        break
            
            h_curr = best_h_local
        
        # Evaluate true score
        h_final = _clip_nonneg(h_curr)
        true_score = compute_score(h_final)
        
        if true_score < best_score:
            best_score = true_score
            best_h = h_final.copy()
    
    # Final coordinate descent refinement using FFT convolution
    for outer in range(5):
        if time.time() > deadline - 5:
            break
        
        # Try reducing peaks: find where autoconvolution is largest
        conv = _fft_convolve(best_h)
        k_peak = np.argmax(conv)
        
        # For indices that contribute most to the peak, try reducing them
        # conv[k_peak] = sum_i h[i] * h[k_peak - i]
        contributing = np.zeros(N)
        for i in range(N):
            j = k_peak - i
            if 0 <= j < N:
                contributing[i] = best_h[i] * best_h[j]
        
        # Try reducing top contributors
        top_indices = np.argsort(contributing)[-20:]
        for i in top_indices:
            if time.time() > deadline - 5:
                break
            if best_h[i] < 1e-8:
                continue
            
            h_new = best_h.copy()
            h_new[i] *= 0.7
            s_new = np.sum(h_new)
            if s_new < 1e-12:
                continue
            conv_new = _fft_convolve(h_new)
            score_new = np.max(conv_new) / (s_new * s_new)
            
            if score_new < best_score - 1e-12:
                best_score = score_new
                best_h = h_new
        
        # Try adding to low areas
        low_indices = np.argsort(best_h)[:10]
        for i in low_indices:
            if time.time() > deadline - 5:
                break
            if best_h[i] > 0.5:
                continue
            
            h_new = best_h.copy()
            h_new[i] += 0.1
            s_new = np.sum(h_new)
            conv_new = _fft_convolve(h_new)
            score_new = np.max(conv_new) / (s_new * s_new)
            
            if score_new < best_score - 1e-12:
                best_score = score_new
                best_h = h_new
    
    return best_h.tolist()
