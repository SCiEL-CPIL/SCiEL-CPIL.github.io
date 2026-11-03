"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Convex optimization in the Fourier domain.
1. Represent the step function f by its DFT coefficients F.
2. The autoconvolution c = f * f has DFT C = |F|^2.
3. We want to minimize max(c) subject to f >= 0 and sum(f) = 1.
4. Use a smooth surrogate (log-sum-exp) for the max of the time-domain autoconvolution.
5. However, computing the time-domain max from F is expensive.
   Instead, we use the fact that we can compute c from F via IFFT.
   To make it differentiable and convex, we optimize over F directly.
   Wait, the constraint f >= 0 is non-convex in F (it's a cone constraint).
   
   Alternative: Optimize over f directly but use FFT for fast gradient evaluation.
   The objective J(f) = max(f*f) / (sum(f))^2 is not convex.
   But we can use a smooth approximation: J_smooth(f) = (1/beta) * log(sum(exp(beta * (f*f)))) / (sum(f))^2.
   This is still non-convex due to the product f*f.
   
   Let's stick to the time-domain but use a much more efficient optimizer:
   We will use a projected gradient descent on the log-sum-exp surrogate.
   The gradient of the log-sum-exp max of autoconvolution can be computed efficiently using FFT.
   
   Let g = f * f.
   L = (1/beta) log(sum(exp(beta * g)))
   dL/dg_k = exp(beta * g_k) / sum(exp(beta * g_j))
   dL/df_i = sum_k (dL/dg_k) * (dg_k/df_i)
   g_k = sum_j f_j f_{k-j}
   dg_k/df_i = f_{k-i} + f_{i-k} (if we consider full convolution symmetry)
   Actually, for autoconvolution of a real sequence:
   g_k = sum_j f_j f_{k-j}
   dg_k/df_i = f_{k-i} + f_{i-k} (with appropriate indexing/boundary handling)
   
   This allows O(N log N) gradient computation per step.
   We will use this for a fast gradient descent with adaptive step size.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill

def fft_convolve(f, g=None):
    """FFT-based convolution for speed."""
    if g is None:
        g = f
    n = len(f) + len(g) - 1
    size = 1
    while size < n:
        size *= 2
    F = np.fft.rfft(f, size)
    G = np.fft.rfft(g, size)
    return np.fft.irfft(F * G, size)[:n]

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = fft_convolve(f, f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def objective_and_grad(f, beta=50.0):
    """
    Compute the log-sum-exp surrogate for the max of autoconvolution
    and its gradient with respect to f.
    
    Returns:
        val: value of the surrogate (scaled by 1/s^2)
        grad: gradient of the surrogate (scaled by 1/s^2)
    """
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf, np.zeros_like(f)
    
    # Normalize f to sum to 1 for stable computation of the shape
    # The score is scale-invariant, so we can optimize the shape
    # Let h = f / s. Then score = max(h*h).
    # We optimize h directly with sum(h)=1.
    
    h = f / s
    
    conv = fft_convolve(h, h)
    
    # Log-sum-exp for stability
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 0.0, np.zeros_like(f)
    
    shifted = conv - max_conv
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum_exp = max_conv + np.log(sum_exp) / beta
    
    # Probability weights for each lag
    weights = exp_shifted / sum_exp
    
    # Gradient of log_sum_exp w.r.t. conv is 'weights'
    # d(log_sum_exp)/d(h_i) = sum_k weights[k] * d(conv_k)/d(h_i)
    # conv_k = sum_j h_j h_{k-j}
    # d(conv_k)/d(h_i) = h_{k-i} + h_{i-k} (for valid indices)
    
    grad_h = np.zeros(N)
    
    # Efficient gradient computation using convolution
    # grad_h[i] = sum_k weights[k] * (h[k-i] + h[i-k])
    # This is a convolution of weights with h, plus weights with h flipped.
    # Let W = weights.
    # Term 1: sum_k W[k] h[k-i] -> This is (W * h)[i] but with reversed indices?
    # Let's check: (W * h)[i] = sum_k W[k] h[i-k].
    # We need sum_k W[k] h[k-i]. Let j = k-i => k = j+i.
    # sum_j W[j+i] h[j]. This is cross-correlation.
    # Cross-correlation of W and h can be computed as convolution of W and flipped h.
    
    # Let's use FFT for this too.
    # grad_h[i] = sum_k W[k] h[k-i] + sum_k W[k] h[i-k]
    # The second term is (W * h)[i] (standard convolution).
    # The first term is (W * h_rev)[i] where h_rev is h reversed?
    # Actually, since h is real, and we are dealing with full convolution lengths,
    # it's easier to just do it with FFTs of the right sizes.
    
    # Simpler: Use the property that the gradient is related to the convolution of weights with h.
    # Let's compute the gradient in the frequency domain.
    # G = FFT(conv) = |FFT(h)|^2
    # This path is tricky for the exact gradient of the LSE surrogate.
    
    # Let's stick to direct computation but optimized.
    # N=600, so N^2 is 360,000. This is very fast in Python/Numpy if vectorized.
    # However, we are doing this many times.
    # Let's use the FFT trick for the gradient.
    
    # grad_h = ifft( fft(weights) * (fft(h) + fft(h[::-1])) ) ... wait, indices are tricky.
    
    # Let's just do the O(N^2) loop in NumPy vectorized form if it's fast enough.
    # 360k operations is negligible.
    
    # Construct the matrix of h shifted? No, that's memory heavy.
    # Let's use the convolution definition.
    # grad_h[i] = sum_k weights[k] * (h[(k-i) mod N] + h[(i-k) mod N]) ?
    # No, the convolution is linear, not circular.
    
    # Let's use the FFT method for the gradient of the LSE of convolution.
    # This is a known technique.
    # Let A = weights.
    # grad_h = A * h + A * h_flipped (circular convolution)
    # Since we are using linear convolution for the autoconvolution, we must be careful.
    # The autoconvolution of length 2N-1.
    # The gradient w.r.t. h_i affects conv_k for k in [i, i+N-1] and [0, N-1-i] etc.
    
    # Given N=600, let's just do the direct sum. It's fast.
    # grad_h[i] = sum_{k=0}^{2N-2} weights[k] * (h[k-i] if 0<=k-i<N else 0 + h[i-k] if 0<=i-k<N else 0)
    
    # Vectorized:
    # Create an array of indices
    # This is effectively a cross-correlation.
    
    # Let's use FFT for cross-correlation.
    size = 1
    while size < 2*N:
        size *= 2
        
    W_fft = np.fft.rfft(weights, size)
    H_fft = np.fft.rfft(h, size)
    
    # Term 1: sum_k W[k] h[k-i]
    # This is cross-correlation.
    # corr[i] = sum_k W[k] h[k-i]
    # In frequency domain: corr = ifft( W_fft * conj(H_fft) )
    
    corr1 = np.fft.irfft(W_fft * np.conj(H_fft), size)[:N]
    
    # Term 2: sum_k W[k] h[i-k]
    # This is standard convolution (W * h)[i]
    conv2 = np.fft.irfft(W_fft * H_fft, size)[:N]
    
    grad_h = corr1 + conv2
    
    # Scale by 1/s^2 because we optimized h = f/s
    # d(score)/df = d(score)/dh * dh/df
    # score = max(h*h). h = f/s.
    # This scaling is complex. Let's just optimize h directly with sum(h)=1.
    # We will project back to f at the end.
    
    return log_sum_exp, grad_h

def optimize_shape(start_time, max_time):
    """
    Optimize the shape of the function h (sum=1, h>=0) to minimize the LSE surrogate.
    Uses projected gradient descent.
    """
    # Initialize with a smooth bump
    x = np.linspace(-1, 1, N)
    h = np.exp(-20 * x**2)
    h = np.maximum(h, 0)
    h /= np.sum(h)
    
    best_h = h.copy()
    best_val = np.inf
    
    beta = 20.0
    step_size = 0.01
    
    for i in range(5000):
        if time.time() - start_time > max_time:
            break
        
        val, grad = objective_and_grad(h, beta)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Projected Gradient Step
        # We want to minimize val subject to h >= 0 and sum(h) = 1.
        # Gradient step: h_new = h - step_size * grad
        # Then project to the simplex.
        
        h_new = h - step_size * grad
        
        # Project to simplex: find h_proj >= 0, sum(h_proj)=1, min ||h_proj - h_new||^2
        # This can be done efficiently.
        # Sort h_new in descending order
        u = np.sort(h_new)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, N + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        h_proj = np.maximum(h_new - theta, 0)
        
        # Check if sum is close to 1
        if np.abs(np.sum(h_proj) - 1) > 1e-8:
             h_proj = h_proj / np.sum(h_proj)
        
        # Adaptive step size
        if val < best_val:
            step_size = min(step_size * 1.05, 0.1)
        else:
            step_size *= 0.8
            
        h = h_proj
        
        # Increase beta slowly to sharpen the max approximation
        if i % 100 == 0 and beta < 100.0:
            beta *= 1.2
            
    return best_h

def construct():
    start_time = time.time()
    
    # Get the optimized shape
    h_opt = optimize_shape(start_time, TIME_LIMIT - 5)
    
    # Scale h_opt to any convenient scale (e.g., sum = N)
    # The score is scale invariant, but the problem asks for "step heights".
    # The evaluator normalizes by (sum(f))^2, so scaling doesn't matter for the score.
    # We return h_opt * N to have "reasonable" heights.
    f = h_opt * N
    
    # Ensure non-negative
    f = np.maximum(f, 0.0)
    
    # Final score check
    # score = compute_score(f)
    
    return [float(x) for x in f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
