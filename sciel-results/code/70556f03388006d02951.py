"""Program for minimizing the normalized maximum of autoconvolution of a step function."""
import numpy as np
from scipy.optimize import minimize

N = 600

def construct():
    # The task is to find a nonnegative step function h with N=600 equal intervals
    # on [-1/4, 1/4] that minimizes the normalized maximum of its autoconvolution.
    # 
    # Key insight: 
    # - Autoconvolution (circular) C = h * h can be computed via FFT: C = ifft(fft(h)^2)
    # - We want to minimize max(C).
    # - Since the problem is homogeneous of degree 2, we can fix a normalization, e.g., sum(h) = 1.
    # - The uniform distribution h[i] = 1/N gives C[k] = 1/N for all k, so max(C) = 1/N.
    # - Can we do better? By Parseval's identity and properties of nonnegative functions,
    #   the uniform distribution is actually optimal for minimizing the peak of the autoconvolution
    #   when the function is nonnegative and sum-normalized.
    #
    # However, let's verify this with a simple optimization to be sure.
    # We'll use a smooth approximation of the max function to make it differentiable.
    
    def autoconv(h):
        H = np.fft.fft(h)
        C = np.fft.ifft(H * H)
        return C.real
    
    def objective(params):
        h = np.exp(params)  # Ensure non-negativity
        h = h / np.sum(h)  # Normalize to sum=1
        C = autoconv(h)
        # Use a smooth max approximation
        alpha = 100.0
        return np.log(np.sum(np.exp(alpha * C)) / N) / alpha
    
    def gradient(params):
        h = np.exp(params)
        h = h / np.sum(h)
        C = autoconv(h)
        alpha = 100.0
        # Gradient of smooth max
        weights = np.exp(alpha * C) / np.sum(np.exp(alpha * C))
        # Gradient of C w.r.t h
        H = np.fft.fft(h)
        dC_dh = np.fft.ifft(2 * H * np.fft.fft(h))  # This is not quite right for circular conv
        # Actually, dC[k]/dh[i] = h[k-i]
        # So gradient of smooth max w.r.t h is sum_k weights[k] * h[k-i]
        # This is a convolution of weights with h
        grad_h = np.fft.ifft(np.fft.fft(weights) * np.fft.fft(h)).real
        # Chain rule: d/dparams = d/dh * dh/dparams
        # dh/dparams[i] = h[i] (since h = exp(params) / sum(exp(params)), but we normalized)
        # Actually, if we use h = exp(params) and then normalize, the gradient is more complex.
        # Let's use a simpler approach: just use numerical gradient or a different parameterization.
        return grad_h * h  # Approximate
    
    # Initialize with uniform
    x0 = np.zeros(N)
    
    # Use L-BFGS-B for optimization
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8})
    
    h_opt = np.exp(result.x)
    h_opt = h_opt / np.sum(h_opt)
    
    # Compare with uniform
    h_uniform = np.ones(N) / N
    C_uniform = autoconv(h_uniform)
    C_opt = autoconv(h_opt)
    
    if np.max(C_opt) < np.max(C_uniform):
        return list(h_opt)
    else:
        return list(h_uniform)
