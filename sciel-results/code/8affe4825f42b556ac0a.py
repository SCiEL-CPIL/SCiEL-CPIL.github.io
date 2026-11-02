"""
Strategy: Spectral initialization + Nesterov accelerated gradient descent.
Minimizes the normalized peak of the autoconvolution for a nonnegative step function.
"""
import time
import numpy as np

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

def compute_score_and_grad(h, beta=50.0):
    """
    Compute smooth approximation of the score and its gradient.
    Uses log-sum-exp for smooth max.
    """
    s = np.sum(h)
    if s == 0:
        return 0.0, np.zeros_like(h)
    
    # Pad h for FFT-based convolution
    n_conv = 2 * N - 1
    n_fft = 1 << (n_conv - 1).bit_length()
    
    # Zero-pad h
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h
    
    # FFT
    H = np.fft.rfft(h_pad)
    H2 = H * H
    conv = np.fft.irfft(H2, n=n_fft)[:n_conv]
    
    # Smooth max using log-sum-exp
    max_conv = np.max(conv)
    # Soft-max: sum(exp(beta*(conv - max_conv))) / beta
    shifted = conv - max_conv
    logsumexp = np.log(np.sum(np.exp(beta * shifted))) / beta
    smooth_max = max_conv + logsumexp
    
    score = smooth_max / (s * s)
    
    # Gradient of smooth_max w.r.t. h
    # d(smooth_max)/d(conv[k]) = exp(beta*(conv[k] - max_conv)) / sum(exp(beta*(conv[j] - max_conv)))
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    
    # Compute gradient efficiently using FFT
    # d(conv[k])/d(h[i]) = h[k-i] + h[i-k] (symmetric)
    # This is equivalent to 2 * (weights * h) convolved appropriately
    
    # Use FFT to compute the gradient
    W = np.fft.rfft(weights, n=n_fft)
    W2 = W * H
    grad_smooth_max = 2 * np.fft.irfft(W2, n=n_fft)[:N]
    
    # Adjust for the fact that we need to be careful about the convolution
    # Actually, let's verify: conv[k] = sum_j h[j] * h[k-j]
    # d(conv[k])/d(h[i]) = h[k-i] + h[i-k] (when indices are valid)
    # sum_k weights[k] * d(conv[k])/d(h[i]) = sum_k weights[k] * (h[k-i] + h[i-k])
    # This is 2 * sum_k weights[k] * h[k-i] (by symmetry)
    # Which is 2 * (weights * h)[i] (convolution of weights and h at lag i)
    
    # So grad_smooth_max = 2 * (weights * h)[i]
    # Let's compute this properly
    grad_smooth_max = np.zeros(N)
    for i in range(N):
        # (weights * h)[i] = sum_k weights[k] * h[i-k]
        # where k ranges such that both indices are valid
        # This is a cross-correlation
        pass
    
    # Let's use the FFT approach but be more careful
    # Actually, the FFT approach I had should work. Let me just use it.
    
    grad_score = grad_smooth_max / (s * s) - 2 * score / s
    
    return score, grad_score

def _spectral_init():
    """
    Initialize using spectral method: 
    We want h such that |H|^2 is as flat as possible (to minimize peak of autoconvolution).
    This means |H| should be roughly constant where H is nonzero.
    Start with a flat spectrum in the low-frequency band.
    """
    x = np.linspace(-0.25, 0.25, N)
    
    # Try various parametric forms
    best_h = np.ones(N)
    best_score = compute_score(best_h)
    
    # 1. Cosine window
    h_cos = 0.5 * (1 + np.cos(4 * np.pi * x))
    s = compute_score(h_cos)
    if s < best_score:
        best_score = s
        best_h = h_cos.copy()
    
    # 2. Gaussian
    for sigma in [0.05, 0.08, 0.1, 0.12, 0.15]:
        h_g = np.exp(-x**2 / (2 * sigma**2))
        s = compute_score(h_g)
        if s < best_score:
            best_score = s
            best_h = h_g.copy()
    
    # 3. Power bump
    for p in [0.5, 1, 2, 3, 4, 5]:
        t = x / 0.25
        h_p = np.maximum(0, 1 - t**2) ** p
        s = compute_score(h_p)
        if s < best_score:
            best_score = s
            best_h = h_p.copy()
    
    # 4. Spectral init: flat spectrum in band
    # We want H(f) to have constant magnitude in some band
    n_fft = 1 << (2 * N - 2).bit_length()
    H_target = np.zeros(n_fft // 2 + 1)
    
    # Try different bandwidths
    for bw_frac in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        bw = int(bw_frac * n_fft // 2)
        H_target[:bw] = 1.0
        h_spec = np.fft.irfft(H_target, n=n_fft)[:N]
        h_spec = np.maximum(0, h_spec)
        if np.sum(h_spec) > 0:
            h_spec = h_spec / np.sum(h_spec)
            s = compute_score(h_spec)
            if s < best_score:
                best_score = s
                best_h = h_spec.copy()
    
    return best_h, best_score

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Spectral initialization
    best_h, best_score = _spectral_init()
    
    if time.time() > deadline:
        return best_h.tolist()
    
    # Nesterov accelerated gradient descent
    h = best_h.copy()
    h = h / np.sum(h)  # Normalize
    
    v = np.zeros_like(h)
    lr = 0.005
    beta = 30.0
    
    for iteration in range(3000):
        if time.time() > deadline:
            break
        
        # Nesterov: evaluate gradient at h - v
        h_nest = h - lr * v
        h_nest = np.maximum(0, h_nest)
        if np.sum(h_nest) > 0:
            h_nest = h_nest / np.sum(h_nest)
        
        score, grad = compute_score_and_grad(h_nest, beta)
        
        # Update velocity
        v_new = 0.9 * v - lr * grad
        
        # Update position
        h_new = h - v_new
        h_new = np.maximum(0, h_new)
        
        s = np.sum(h_new)
        if s > 0:
            h_new = h_new / s
        
        new_score = compute_score(h_new)
        
        if new_score < best_score:
            best_score = new_score
            best_h = h_new.copy()
            h = h_new.copy()
            v = v_new.copy()
        else:
            # Backtracking line search
            step = 1.0
            for _ in range(5):
                step *= 0.5
                h_trial = h - step * v_new
                h_trial = np.maximum(0, h_trial)
                s_t = np.sum(h_trial)
                if s_t > 0:
                    h_trial = h_trial / s_t
                    score_trial = compute_score(h_trial)
                    if score_trial < best_score:
                        best_score = score_trial
                        best_h = h_trial.copy()
                        h = h_trial.copy()
                        v = step * v_new.copy()
                        break
            else:
                # If no improvement, just update velocity
                v = v_new.copy() * 0.5
        
        # Anneal beta periodically
        if iteration % 100 == 0 and beta < 200:
            beta *= 1.3
        
        # Reduce learning rate slowly
        if iteration % 200 == 0:
            lr *= 0.95
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
