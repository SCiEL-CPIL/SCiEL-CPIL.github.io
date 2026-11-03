"""Optimized program for first autocorrelation inequality upper bound."""
import numpy as np
import time

N = 600

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Full linear autoconvolution via FFT
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)
    conv = conv[:size]
    return np.max(conv) / (s * s)

def project_simplex(v, s=1.0):
    """Project v onto the simplex {x | x >= 0, sum(x) = s}."""
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    cond = u * np.arange(1, n + 1) > (cssv - s)
    rho = np.nonzero(cond)[0]
    if len(rho) == 0:
        return np.zeros(n)
    rho = rho[-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    # Ensure exact sum
    wsum = np.sum(w)
    if wsum > 1e-15:
        w = w * (s / wsum)
    return w

def smooth_max_gradient(f, beta):
    """
    Compute gradient of smooth_max(autoconv(f)) where smooth_max uses log-sum-exp.
    Returns (smooth_max_value, gradient_array).
    """
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)
    conv = conv[:size]
    
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 0.0, np.zeros(N)
    
    z = beta * (conv - max_conv)
    exp_z = np.exp(z)
    sum_exp = np.sum(exp_z)
    if sum_exp < 1e-300:
        return max_conv, np.zeros(N)
    p = exp_z / sum_exp  # weights on each lag
    
    # Gradient: grad_i = sum_k p_k * (d conv_k / d f_i)
    # conv_k = sum_{j=0}^{N-1-k} f_j f_{j+k} for k >= 0
    # d conv_k / d f_i = f_{i+k} (if i+k < N) + f_{i-k} (if i-k >= 0)
    # 
    # grad_i = sum_{k=0}^{N-1-i} p_k f_{i+k} + sum_{k=0}^{i} p_k f_{i-k}
    # 
    # First term: for fixed i, sum over k of p_k * f_{i+k} = (p * f)_i in correlation sense
    # Actually: first term = sum_{k=0}^{N-1-i} p_k * f[i+k]
    # This is like a cross-correlation of p and f, but only for valid indices.
    #
    # Let's compute efficiently using FFT or direct vectorized ops.
    # For N=600, we can do this with matrix operations or loops.
    
    # Vectorized approach using Toeplitz-like structure
    # First part: A_i = sum_{k=0}^{N-1-i} p_k * f_{i+k}
    # Second part: B_i = sum_{k=0}^{i} p_k * f_{i-k}
    
    # A_i = sum_{j=i}^{N-1} p_{j-i} * f_j  (let j = i+k, k = j-i)
    # B_i = sum_{j=0}^{i} p_{i-j} * f_j
    
    # We can compute these using convolution:
    # A = ifft(fft(p_padded) * fft(f_padded)) but with linear conv...
    # Simpler: use np.correlate or explicit computation
    
    # For efficiency with N=600, let's use a matrix-free approach with FFT
    # A_i = sum_{j=i}^{N-1} p_{j-i} f_j = (p * f) where * is cross-correlation restricted
    
    # Let's just do it with a loop but vectorized inner product
    grad = np.zeros(N)
    for i in range(N):
        # Part 1: k from 0 to N-1-i
        if N - 1 - i >= 0:
            grad[i] += np.dot(p[0:N-i], f[i:N])
        # Part 2: k from 0 to i  
        if i >= 0:
            grad[i] += np.dot(p[0:i+1], f[i::-1])
    
    return max_conv + np.log(sum_exp) / beta, grad

def construct():
    deadline = time.time() + 140  # 140s budget, leaving 10s safety margin
    
    # Initial candidates
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5/N
    candidates = []
    
    # Box (seed)
    candidates.append(np.ones(N))
    
    # Gaussian-like peaks
    for sigma in [0.05, 0.08, 0.1, 0.12, 0.15]:
        g = np.exp(-0.5 * (x/sigma)**2)
        candidates.append(g)
    
    # Cosine
    c = 0.5 * (1 + np.cos(2*np.pi*x))
    candidates.append(c)
    
    # Triangle
    t = np.maximum(0, 1 - np.abs(x)/0.125)
    candidates.append(t)
    
    # Find best initial
    best_f = None
    best_score = np.inf
    for cand in candidates:
        f_norm = cand / np.sum(cand)
        s = compute_score(f_norm)
        if s < best_score:
            best_score = s
            best_f = f_norm.copy()
    
    f = best_f.copy()
    
    # Smooth-max projected gradient descent
    beta = 5.0  # temperature for smooth max
    lr = 0.01
    best_f = f.copy()
    best_val = best_score
    
    for it in range(300):
        if time.time() > deadline - 10:
            break
        
        smooth_val, grad = smooth_max_gradient(f, beta)
        
        # Normalize gradient by current scale
        f_new = f - lr * grad
        
        # Project onto simplex {x >= 0, sum(x) = 1}
        f_new = project_simplex(f_new, 1.0)
        
        # Check improvement on actual objective
        val = compute_score(f_new)
        if val < best_val:
            best_val = val
            best_f = f_new.copy()
            f = f_new
            # Slowly increase beta to sharpen the max approximation
            if it % 30 == 0 and beta < 50.0:
                beta *= 1.5
        else:
            lr *= 0.5
            if lr < 1e-8:
                break
    
    # Final polish with smaller beta for finer optimization
    if time.time() < deadline - 20:
        beta = 20.0
        lr = 0.005
        for it in range(100):
            if time.time() > deadline - 10:
                break
            smooth_val, grad = smooth_max_gradient(f, beta)
            f_new = f - lr * grad
            f_new = project_simplex(f_new, 1.0)
            val = compute_score(f_new)
            if val < best_val:
                best_val = val
                best_f = f_new.copy()
                f = f_new
            else:
                lr *= 0.5
                if lr < 1e-9:
                    break
    
    return list(best_f)
