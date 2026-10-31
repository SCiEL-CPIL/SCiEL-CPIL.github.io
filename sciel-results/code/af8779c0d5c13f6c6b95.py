"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Model as discrete probability distribution and use gradient descent
with projection onto the probability simplex to minimize the normalized max
of the autoconvolution. Use smooth surrogate (soft-max via log-sum-exp) for
the max operation to get differentiable objective.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def soft_max(conv, temperature=1.0):
    """Smooth approximation of max using log-sum-exp."""
    scaled = conv / temperature
    max_val = np.max(scaled)
    if max_val > 500:  # numerical stability
        return max_val * temperature
    return temperature * (max_val + np.log(np.sum(np.exp(scaled - max_val))))

def compute_soft_score(f, temperature=1.0):
    """Compute smooth normalized max of autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return soft_max(conv, temperature) / (s * s)

def project_simplex(v):
    """Project vector onto the probability simplex (nonnegative, sums to 1)."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    rho = np.max(ind[cond])
    theta = cssv[rho - 1] / rho
    w = v - theta
    return np.maximum(w, 0)

def gradient_soft_score(f, temperature=1.0):
    """
    Compute gradient of soft normalized max of autoconvolution.
    f is a probability vector (sums to 1, nonnegative).
    
    The objective is: soft_max(conv) / (sum(f))^2
    Since we're on the simplex, sum(f) = 1, so objective is just soft_max(conv).
    """
    f = np.asarray(f, dtype=np.float64)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    
    # Soft max gradient: softmax(conv / temperature) * (1/temperature)
    scaled = conv / temperature
    max_val = np.max(scaled)
    exp_scaled = np.exp(scaled - max_val)
    softmax_weights = exp_scaled / np.sum(exp_scaled)
    
    # Gradient of soft_max(conv) w.r.t. conv is softmax_weights / temperature
    grad_conv = softmax_weights / temperature
    
    # conv[k] = sum_i f[i] * f[k-i]
    # d(conv[k])/d(f[j]) = f[k-j] + f[j-k] (for valid indices)
    # This is the same as convolve(f, grad_conv) in a specific way
    
    # Actually, for autoconvolution: conv = f * f (full convolution)
    # d(conv)/df is a linear operator. We can compute grad_f such that
    # grad_f[j] = sum_k grad_conv[k] * d(conv[k])/d(f[j])
    # d(conv[k])/d(f[j]) = f[k-j] if 0 <= k-j < N, else 0
    # So grad_f = convolve(f, grad_conv, mode='same') adjusted
    
    # More precisely: conv[k] = sum_{i} f[i] * f[k-i] where k ranges 0 to 2N-2
    # d(conv[k])/d(f[j]) = f[k-j] + f[j-k] (but only one term is nonzero for each k,j)
    # Actually: if we fix j, then conv[k] depends on f[j] when k-j is in [0,N-1] or j-k is in [0,N-1]
    # So d(conv[k])/d(f[j]) = f[k-j] if 0 <= k-j < N, plus f[j-k] if 0 <= j-k < N
    # But these are the same when we sum over k with grad_conv[k]
    
    # Using FFT for efficiency:
    # The gradient of sum_k grad_conv[k] * conv[k] w.r.t. f is:
    # grad_f = ifft(fft(f) * fft(grad_conv_padded))
    
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    G = np.fft.rfft(grad_conv, n=n_fft)
    grad_f = np.fft.irfft(F * G, n=n_fft)[:N]
    
    # Normalize by sum(f)^2 if not on simplex
    if abs(s - 1.0) > 1e-10:
        grad_f = grad_f / (s * s)
    
    return grad_f

def construct():
    start_time = time.time()
    
    # Initial: uniform distribution (constant function normalized)
    best_f = np.ones(N) / N
    best_score = compute_score(best_f)
    
    # Phase 1: Gradient descent with simplex projection
    current_f = best_f.copy()
    temperature = 1.0  # Start with exact max (high temperature = smooth)
    
    # Use adaptive temperature: start smooth, then sharpen
    temperatures = [5.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05]
    
    for temp in temperatures:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        temperature = temp
        lr = 0.01
        num_steps = 200
        
        for step in range(num_steps):
            if time.time() - start_time > TIME_LIMIT - 20:
                break
            
            # Compute gradient
            grad = gradient_soft_score(current_f, temperature)
            
            # Gradient step
            candidate = current_f - lr * grad
            
            # Project onto simplex
            candidate = project_simplex(candidate)
            
            # Evaluate
            cand_score = compute_soft_score(candidate, temperature)
            exact_score = compute_score(candidate)
            
            if cand_score < compute_soft_score(current_f, temperature):
                current_f = candidate
                if exact_score < best_score:
                    best_f = candidate.copy()
                    best_score = exact_score
            else:
                # Reduce learning rate
                lr *= 0.9
    
    # Phase 2: Local refinement with smaller perturbations
    if time.time() - start_time < TIME_LIMIT - 10:
        current_f = best_f.copy()
        current_score = best_score
        lr = 0.001
        
        for iteration in range(100):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            # Small random perturbation
            perturbation = np.random.randn(N) * lr
            candidate = current_f + perturbation
            candidate = project_simplex(candidate)
            
            cand_score = compute_score(candidate)
            
            if cand_score < current_score:
                current_f = candidate
                current_score = cand_score
                if current_score < best_score:
                    best_f = current_f.copy()
                    best_score = current_score
    
    # Scale back to original scale (the score is scale-invariant for normalized max)
    # But we need to return nonnegative step heights. The normalization in compute_score
    # divides by (sum(f))^2, so scaling doesn't change the score.
    # Return as-is (probability vector) or scale to have reasonable magnitudes.
    # Let's scale so that max is around 1 for numerical stability
    max_val = np.max(best_f)
    if max_val > 0:
        return [float(x / max_val) for x in best_f]
    else:
        return [1.0 / N] * N

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
