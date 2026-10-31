import numpy as np
import time

N = 600
_w = 1.0 / (2.0 * N)

def _autoconv_max(heights):
    """Compute max_t (f*f)(t) / integral(f) for step function f."""
    h = np.asarray(heights, dtype=np.float64)
    integral_f = np.sum(h) * _w
    if integral_f <= 0:
        return float('inf')
    
    # Use FFT to compute autoconvolution
    # Pad to next power of 2 for efficiency
    n = len(h)
    n_fft = 1 << (2 * n - 1).bit_length()
    
    F = np.fft.rfft(h, n=n_fft)
    # Autoconvolution: correlate h with itself
    # The linear autoconvolution at lag k is sum_i h[i]*h[i+k] (circular if not padded)
    # With proper padding, we get linear autoconvolution
    conv = np.fft.irfft(F * F.conj(), n=n_fft)
    # The autoconvolution values: conv[k] for k=0..n-1 gives (f*f)(k*w)
    # Actually, for autoconvolution of a real signal, the result is symmetric
    # conv[0] = sum h[i]^2 * w (peak at zero lag)
    # We need to scale: the actual (f*f)(t) at t=k*w is conv[k] * w
    # But wait, the FFT convolution gives sum of products, so:
    # (f*f)(k*w) ≈ w * conv[k]
    
    # The maximum of the autoconvolution
    # For step functions, the max occurs at t=0 (by symmetry and triangle inequality for nonneg)
    # Actually no - for oscillatory functions, the max might not be at 0
    # But for nonnegative functions, (f*f)(0) = integral f^2, and by Cauchy-Schwarz
    # this is >= (integral f)^2 / support_width, but the max could be elsewhere
    
    # Let's just take the max over all lags
    max_autoconv = np.max(conv) * _w
    
    return max_autoconv / integral_f

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    np.random.seed(42)
    
    # Start with uniform
    best_heights = np.ones(N)
    best_score = _autoconv_max(best_heights)
    
    # Strategy: Use a parametric family with alternating high/low values
    # h_i = a + b * cos(2*pi*m*i/N) for various m
    # This creates periodic oscillations that can reduce the autoconvolution peak
    
    # Try different frequencies m
    best_params = (1.0, 0.0, 1)  # a, b, m
    
    for m in range(1, 30):
        if time.time() > deadline - 20:
            break
        
        # For frequency m, try different amplitudes
        # h_i = a + b*cos(2*pi*m*i/N), must be nonnegative
        # So a >= |b|
        
        for a in np.linspace(0.5, 1.5, 10):
            if time.time() > deadline - 10:
                break
            for b in np.linspace(0, a - 0.01, 8):
                if time.time() > deadline - 5:
                    break
                
                i = np.arange(N)
                h = a + b * np.cos(2 * np.pi * m * i / N)
                h = np.maximum(h, 0)
                
                if np.sum(h) <= 0:
                    continue
                
                score = _autoconv_max(h)
                if score < best_score:
                    best_score = score
                    best_heights = h.copy()
                    best_params = (a, b, m)
    
    # Now refine with local search around the best solution
    if time.time() < deadline - 15:
        h = best_heights.copy()
        current_score = best_score
        
        # Gradient-free local search: perturb random subsets
        T = 0.02
        for step in range(3000):
            if time.time() > deadline - 5:
                break
            
            # Perturb a random block of consecutive elements
            block_size = np.random.randint(1, 20)
            start_idx = np.random.randint(0, N - block_size + 1)
            end_idx = start_idx + block_size
            
            delta = np.random.randn() * 0.1
            new_h = h.copy()
            new_h[start_idx:end_idx] += delta
            new_h = np.maximum(new_h, 0)
            
            new_score = _autoconv_max(new_h)
            
            if new_score < current_score or (np.random.rand() < np.exp(-(new_score - current_score) / T)):
                h = new_h
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_heights = h.copy()
            
            T *= 0.998
            if T < 0.001:
                break
    
    return list(best_heights)

if __name__ == "__main__":
    result = construct()
    print(f"Score: {_autoconv_max(result)}")
