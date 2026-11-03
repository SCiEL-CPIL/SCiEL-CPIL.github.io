import numpy as np
import time
import math

N = 600
WIDTH = 0.5  # Total width of the interval
N_INTERVALS = N

# Deadline guard: return best result if time is running out
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    # We need to minimize the "normalized maximum of its autoconvolution"
    # Let f be a step function with N equal intervals on [-1/4, 1/4]
    # Each interval has width h = WIDTH / N = 0.5 / 600
    # f is nonnegative, piecewise constant
    
    # Autoconvolution: (f*f)(t) = integral f(s) f(t-s) ds
    # For step functions, this can be computed exactly
    # The "normalized maximum" likely means max(f*f) / (some normalization)
    # Common normalization: max(f*f) / (integral f)^2 or max(f*f) / max(f) / integral(f)
    # Or perhaps max(f*f) / integral(f^2)
    
    # Let's think about what the score 1.0 means for the constant function
    # If f = c (constant), then f*f has a triangular shape
    # (f*f)(t) for |t| < W: c^2 * (W - |t|)
    # max(f*f) = c^2 * W
    # integral(f) = c * W
    # integral(f)^2 = c^2 * W^2
    # max(f*f) / integral(f)^2 = c^2 * W / (c^2 * W^2) = 1/W = 2
    
    # If normalization is max(f*f) / (max(f) * integral(f)):
    # = c^2 * W / (c * c * W) = 1
    # That gives 1.0 for constant!
    
    # So the metric is likely: max(f*f) / (max(f) * integral(f))
    # We want to minimize this.
    
    # Let's verify: for a step function with heights h_i on intervals of width h:
    # integral(f) = sum(h_i * h) = h * sum(h_i)
    # max(f) = max(h_i)
    # max(f*f) = max over t of autoconvolution
    
    # The autoconvolution of a step function can be computed efficiently
    # Let's try a gradient-based approach to find a better step function
    
    def compute_score(heights):
        """Compute the normalized maximum autoconvolution score"""
        heights = np.asarray(heights, dtype=np.float64)
        h = WIDTH / N_INTERVALS
        # Autoconvolution via FFT for speed
        # Pad to avoid circular convolution
        n_fft = 2 * N_INTERVALS
        f_padded = np.zeros(n_fft)
        f_padded[:N_INTERVALS] = heights
        # Linear autoconvolution
        F = np.fft.rfft(f_padded)
        conv = np.fft.irfft(F * F, n=n_fft)
        max_conv = np.max(conv)
        
        integral_f = np.sum(heights) * h
        max_f = np.max(heights)
        
        if integral_f <= 0 or max_f <= 0:
            return float('inf')
        
        return max_conv / (max_f * integral_f)
    
    # Start with constant function
    best_heights = np.ones(N_INTERVALS)
    best_score = compute_score(best_heights)
    
    # Try some structured patterns
    # 1. Gaussian-like bump
    x = np.linspace(-WIDTH/2, WIDTH/2, N_INTERVALS)
    for sigma in [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]:
        heights = np.exp(-x**2 / (2 * sigma**2))
        heights = np.maximum(heights, 0)
        score = compute_score(heights)
        if score < best_score:
            best_score = score
            best_heights = heights.copy()
    
    # 2. Cosine bump
    for freq in [1, 2, 3, 4, 5]:
        heights = 0.5 * (1 + np.cos(2 * np.pi * freq * x / WIDTH))
        heights = np.maximum(heights, 0)
        score = compute_score(heights)
        if score < best_score:
            best_score = score
            best_heights = heights.copy()
    
    # 3. Try to optimize using gradient descent on a low-dimensional representation
    # Use Fourier basis: f(x) = sum_{k=0}^{K} a_k * cos(k*pi*x/WIDTH) + b_k * sin(k*pi*x/WIDTH)
    # But we need f >= 0
    
    K = 15
    # Initialize with the best so far, project to Fourier coefficients
    def fourier_to_heights(coeffs):
        x = np.linspace(-WIDTH/2, WIDTH/2, N_INTERVALS, endpoint=False)
        f = np.zeros(N_INTERVALS)
        for k in range(len(coeffs) // 2 + 1):
            a = coeffs[2*k]
            b = coeffs[2*k+1] if 2*k+1 < len(coeffs) else 0
            f += a * np.cos(2 * np.pi * k * x / WIDTH)
            if b != 0:
                f += b * np.sin(2 * np.pi * k * x / WIDTH)
        return np.maximum(f, 0)
    
    # Get initial coefficients from best_heights
    x = np.linspace(-WIDTH/2, WIDTH/2, N_INTERVALS, endpoint=False)
    n_coeffs = 2 * K + 1
    coeffs = np.zeros(n_coeffs)
    coeffs[0] = np.mean(best_heights)
    for k in range(1, K+1):
        coeffs[2*k] = 2.0 / N_INTERVALS * np.sum(best_heights * np.cos(2 * np.pi * k * x / WIDTH))
        coeffs[2*k+1] = 2.0 / N_INTERVALS * np.sum(best_heights * np.sin(2 * np.pi * k * x / WIDTH))
    
    # Gradient descent
    lr = 0.01
    for iteration in range(2000):
        if time.time() - start_time > DEADLINE:
            break
        heights = fourier_to_heights(coeffs)
        if np.max(heights) <= 0:
            break
        score = compute_score(heights)
        if score < best_score:
            best_score = score
            best_heights = heights.copy()
        
        # Numerical gradient
        grad = np.zeros(n_coeffs)
        for j in range(n_coeffs):
            coeffs[j] += lr * 0.01
            h_plus = fourier_to_heights(coeffs)
            s_plus = compute_score(h_plus)
            coeffs[j] -= lr * 0.01
            coeffs[j] -= lr * 0.01
            h_minus = fourier_to_heights(coeffs)
            s_minus = compute_score(h_minus)
            coeffs[j] += lr * 0.01
            grad[j] = (s_plus - s_minus) / (2 * lr * 0.01)
        
        coeffs -= lr * grad
    
    # Final check
    final_heights = best_heights
    final_score = best_score
    
    # Ensure nonnegative and return as list
    final_heights = np.maximum(final_heights, 0.0)
    return list(final_heights)

