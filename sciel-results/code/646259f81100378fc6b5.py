import time
import numpy as np
from scipy import optimize

N = 600
FFT_SIZE = 2048

def _fft_convolve(h):
    """Compute linear autoconvolution using FFT for speed."""
    H = np.fft.rfft(h, FFT_SIZE)
    P = H * np.conj(H)
    conv = np.fft.irfft(P, FFT_SIZE)
    return conv[:2*N-1]

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    conv = _fft_convolve(h)
    total = np.sum(h)
    if total == 0:
        return 0.0
    return np.max(conv) / (total * total)

def _logsumexp_surrogate(conv, T):
    """Stable log-sum-exp approximation of max."""
    conv_max = np.max(conv)
    shifted = (conv - conv_max) / T
    shifted = np.clip(shifted, -50, 50)
    return conv_max + T * np.log(np.sum(np.exp(shifted)))

def _objective_and_grad(h, T):
    """Compute log-sum-exp objective and gradient for projected gradient descent."""
    conv = _fft_convolve(h)
    total = np.sum(h)
    if total < 1e-10:
        return 1e10, np.zeros_like(h)
    
    soft_max = _logsumexp_surrogate(conv, T)
    obj = soft_max / (total * total)
    
    # Gradient: d/dh [soft_max / total^2]
    # = (grad_soft_max * total^2 - soft_max * 2 * total * ones) / total^4
    # = grad_soft_max / total^2 - 2 * soft_max / total^3
    
    # grad_soft_max: derivative of log-sum-exp
    weights = np.exp((conv - conv_max) / T)
    weights = np.clip(weights, 0, np.inf)
    weight_sum = np.sum(weights)
    if weight_sum < 1e-10:
        weight_sum = 1e-10
    grad_soft_max = weights / weight_sum  # This is the softmax weights
    
    # Need to map gradient from conv domain back to h domain
    # conv[k] = sum_i h[i] * h[k-i]
    # d conv[k] / d h[j] = h[k-j] + h[k-N+1-j] (for linear conv with zero padding)
    # This is expensive. Instead, use FFT-based gradient computation.
    
    # Alternative: use the fact that conv = ifft(fft(h) * conj(fft(h)))
    # d conv / d h can be computed via FFT
    
    H = np.fft.rfft(h, FFT_SIZE)
    # conv = irfft(H * conj(H))
    # d conv / d h[j] involves the derivative of H with respect to h
    
    # Simpler: use numerical gradient or a different approach
    # For now, let's use a simpler gradient approximation
    
    # Actually, let's use the chain rule properly
    # conv = ifft(|H|^2) where H = fft(h)
    # d conv[k] / d h[j] = Re[ ifft( 2 * H * dH/dh[j] ) ]
    # dH[m] / dh[j] = exp(-2*pi*i*m*j/FFT_SIZE) for m in 0..FFT_SIZE/2
    
    # This is O(N^2) which is too slow. Let's use a different strategy.
    
    # Instead, let's optimize in the frequency domain
    # Minimize soft_max / total^2 subject to h >= 0
    # In frequency domain: H = fft(h), |H|^2 >= 0, h = ifft(H) >= 0
    
    # For projected gradient, we need the gradient in h-space
    # Let's compute it via FFT
    
    # d soft_max / d conv[k] = weights[k] / weight_sum
    # d conv[k] / d h[j] = sum over the convolution structure
    
    # Efficient computation:
    # grad_h = ifft( 2 * H * fft(grad_conv) ) where grad_conv = weights / weight_sum
    grad_conv = grad_soft_max
    G = np.fft.rfft(grad_conv, FFT_SIZE)
    grad_h_fft = 2 * H * G
    grad_h = np.fft.irfft(grad_h_fft, FFT_SIZE)[:N]
    
    grad_obj = grad_h / (total * total) - 2 * soft_max / (total ** 3)
    
    return obj, grad_obj

def _projected_gradient_descent(h0, T, max_iter=2000, tol=1e-12, time_limit=None, start_time=None):
    """Projected gradient descent with backtracking line search."""
    h = h0.copy()
    h = np.maximum(h, 0)
    
    if start_time is None:
        start_time = time.time()
    
    prev_obj = _objective_and_grad(h, T)[0]
    
    for it in range(max_iter):
        if time_limit is not None and time.time() > start_time + time_limit:
            break
        
        obj, grad = _objective_and_grad(h, T)
        
        if abs(prev_obj - obj) < tol:
            break
        prev_obj = obj
        
        # Backtracking line search
        step = 1.0
        for _ in range(20):
            h_new = np.maximum(h - step * grad, 0)
            obj_new = _objective_and_grad(h_new, T)[0]
            if obj_new < obj:
                break
            step *= 0.5
        
        h = h_new
    
    return h, _objective_and_grad(h, T)[0]

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, leaving 10s margin
    
    # Initial guess: all ones
    h = np.ones(N)
    best_h = h.copy()
    best_score = compute_autoconv_max(h)
    
    # Generate diverse initial guesses
    np.random.seed(42)
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N))
    
    # 2. Random positive
    inits.append(np.random.rand(N) + 0.1)
    
    # 3. Cosine window
    t = np.linspace(0, 1, N, endpoint=False)
    inits.append(0.5 * (1 + np.cos(2 * np.pi * t)))
    
    # 4. Two-tone
    h_two = 1.0 + 0.5 * np.cos(2 * np.pi * 5 * t)
    inits.append(np.maximum(h_two, 0.1))
    
    # 5. Random with different seeds
    for seed in [1, 2, 3, 4, 5]:
        np.random.seed(seed)
        inits.append(np.random.rand(N) + 0.1)
    
    # 6. Edge enhanced
    inits.append(1.0 + 0.5 * np.abs(np.linspace(-1, 1, N)))
    
    # 7. Center enhanced
    inits.append(1.0 + 0.5 * np.exp(-5 * (np.linspace(-1, 1, N))**2))
    
    # 8. Flat spectrum attempt
    h_rand = np.random.randn(N)
    H_rand = np.fft.rfft(h_rand, FFT_SIZE)
    phase = np.angle(H_rand)
    H_flat = np.exp(1j * phase)
    h_flat = np.fft.irfft(H_flat, FFT_SIZE)[:N]
    h_flat = np.abs(h_flat)
    if np.max(h_flat) > 0:
        h_flat = h_flat / np.max(h_flat)
    inits.append(h_flat)
    
    candidates = []
    
    for i, h0 in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        # Normalize for numerical stability
        h0 = h0 / np.sum(h0) * N
        
        try:
            # Stage 1: Coarse with higher T
            h1, _ = _projected_gradient_descent(h0, T=0.1, max_iter=500, time_limit=10)
            candidates.append(h1)
            
            if time.time() > deadline - 20:
                break
            
            # Stage 2: Fine with lower T
            h2, _ = _projected_gradient_descent(h1, T=0.01, max_iter=1000, time_limit=15)
            candidates.append(h2)
            
        except Exception:
            pass
    
    # Evaluate all candidates
    for x in candidates:
        h_new = np.abs(x)
        score = compute_autoconv_max(h_new)
        if score < best_score:
            best_score = score
            best_h = h_new.copy()
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            h_polish, _ = _projected_gradient_descent(best_h, T=0.005, max_iter=1000, time_limit=10)
            h_new = np.abs(h_polish)
            score = compute_autoconv_max(h_new)
            if score < best_score:
                best_score = score
                best_h = h_new.copy()
        except Exception:
            pass
    
    return list(best_h)
