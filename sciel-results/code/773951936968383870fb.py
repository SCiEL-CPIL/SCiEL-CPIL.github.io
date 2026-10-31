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
    
    n = len(h)
    n_fft = 1 << (2 * n - 1).bit_length()
    
    F = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(F * F.conj(), n=n_fft)
    max_autoconv = np.max(conv) * _w
    
    return max_autoconv / integral_f

def _score_normalized(h):
    """
    Score assuming the evaluator normalizes max(h) to 1.
    Returns the value to be minimized.
    """
    h = np.asarray(h, dtype=np.float64)
    max_h = np.max(h)
    if max_h <= 0:
        return float('inf')
    h_norm = h / max_h
    return _autoconv_max(h_norm)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    np.random.seed(42)
    
    # Initialize with a cosine wave
    m = 1
    alpha = 0.5
    i = np.arange(N)
    h = 1.0 + alpha * np.cos(2 * np.pi * m * i / N)
    h = np.maximum(h, 0)
    
    best_h = h.copy()
    best_score = _score_normalized(best_h)
    
    # Optimization loop
    # We will use a simple projected gradient descent or coordinate ascent
    # Since we don't have easy gradients for the max, we can use subgradient or finite differences
    # But 600 variables is large for finite differences.
    
    # Alternative: Optimize parameters of a parametric family first
    # h_i = 1 + sum_k alpha_k cos(2*pi*k*i/N + phi_k)
    
    # Let's try optimizing a few Fourier coefficients
    # h = 1 + a1*cos(2pi*i/N) + a2*cos(4pi*i/N) ...
    
    coeffs = np.zeros(10)
    coeffs[0] = 0.5 # m=1
    coeffs[1] = 0.2 # m=2
    
    def generate_h(coeffs):
        h = np.ones(N)
        for k in range(len(coeffs)):
            m = k + 1
            h += coeffs[k] * np.cos(2 * np.pi * m * np.arange(N) / N)
        return np.maximum(h, 0)
    
    current_coeffs = coeffs.copy()
    current_h = generate_h(current_coeffs)
    current_score = _score_normalized(current_h)
    best_coeffs = current_coeffs.copy()
    best_h = current_h.copy()
    best_score = current_score
    
    # Simple gradient descent on coeffs using finite differences
    # This is slow but robust for 10 params
    step_size = 0.05
    for step in range(500):
        if time.time() > deadline - 20:
            break
        
        improved = False
        for k in range(len(coeffs)):
            # Try increasing
            old_val = current_coeffs[k]
            current_coeffs[k] += step_size
            h_test = generate_h(current_coeffs)
            s_test = _score_normalized(h_test)
            if s_test < current_score:
                current_score = s_test
                current_h = h_test
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
                    best_coeffs = current_coeffs.copy()
            else:
                current_coeffs[k] = old_val
                
            # Try decreasing
            current_coeffs[k] -= 2 * step_size
            h_test = generate_h(current_coeffs)
            s_test = _score_normalized(h_test)
            if s_test < current_score:
                current_score = s_test
                current_h = h_test
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
                    best_coeffs = current_coeffs.copy()
            else:
                current_coeffs[k] = old_val
        
        if not improved:
            step_size *= 0.9
            if step_size < 0.001:
                break
                
    # Final local search on h directly?
    # Maybe just return best_h
    
    return list(best_h)

if __name__ == "__main__":
    result = construct()
    print(f"Score: {_score_normalized(result)}")
