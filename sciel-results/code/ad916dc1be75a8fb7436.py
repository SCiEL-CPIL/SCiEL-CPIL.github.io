"""
Third autocorrelation inequality: minimize the upper bound on C3.
The score is seed/value where value is the raw upper bound (lower is better).
We search over signed step functions with 400 intervals on [-1/4, 1/4].
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 115  # safety margin before 120s kill

def _compute_c3_bound(heights):
    """
    Compute the upper bound on C3 from the autoconvolution of the signed step function.
    
    For a step function f(x) = sum_i h_i * 1_{[x_i, x_{i+1}]}(x) on [-1/4, 1/4],
    the autoconvolution (f * f)(t) = integral f(s) f(t-s) ds.
    
    The C3 bound relates to the L-infinity norm of the autoconvolution.
    Specifically, for the third autocorrelation inequality, the bound is:
        C3 <= max_t |(f*f)(t)|
    
    where f is the signed step function. We compute this via FFT for efficiency.
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    # Interval width
    dx = 0.5 / n  # total length 0.5, n intervals
    
    # Pad h to length 2n-1 for linear convolution
    h_pad = np.zeros(2 * n - 1)
    h_pad[:n] = h
    
    # Linear autoconvolution via FFT
    f_fft = np.fft.fft(h_pad, n=2 * (2 * n - 1))
    conv = np.fft.ifft(f_fft * np.conj(f_fft)).real
    
    # Compute max absolute value
    max_abs = np.max(np.abs(conv))
    c3_bound = dx * max_abs
    
    return c3_bound


def _local_search(heights, max_iter=5000, step_size=0.5):
    """
    Local search: try flipping individual bits (changing sign) to reduce the C3 bound.
    """
    h = np.array(heights, dtype=np.float64)
    current_bound = _compute_c3_bound(h)
    
    # Precompute for efficiency: we'll try flipping each position
    # For efficiency, we can compute the effect of flipping h[i] on the autoconvolution.
    # Flipping h[i] to -h[i] changes the autoconvolution.
    # This is complex, so we'll just recompute for now but limit iterations.
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        best_gain = 0
        best_idx = -1
        
        for i in range(N):
            old_val = h[i]
            h[i] = -old_val
            new_bound = _compute_c3_bound(h)
            h[i] = old_val
            
            gain = current_bound - new_bound
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx >= 0:
            h[best_idx] = -h[best_idx]
            current_bound -= best_gain
            improved = True
    
    return h, current_bound


def _generate_fbm_signal(n, H, seed=42):
    """
    Generate a fractional Brownian motion-like signal with Hurst exponent H.
    We use a simple approach: generate white noise, filter it to have the
    desired power spectrum S(f) ~ |f|^{-(2H+1)}, then take the sign.
    
    For H in (0, 1), lower H gives more anti-correlated (short-range) behavior,
    higher H gives more correlated (long-range) behavior.
    """
    np.random.seed(seed)
    
    # Generate white noise
    noise = np.random.randn(n)
    
    # Compute FFT
    F = np.fft.fft(noise)
    freqs = np.fft.fftfreq(n)
    
    # Apply fractional Brownian motion filter
    # Power spectrum should be proportional to |f|^{-(2H+1)}
    # So amplitude spectrum is proportional to |f|^{-(H+0.5)}
    # Avoid division by zero at f=0
    mask = np.abs(freqs) > 1e-10
    filtered_F = F.copy()
    filtered_F[mask] *= np.abs(freqs[mask]) ** (-(H + 0.5))
    filtered_F[~mask] = 0  # DC component
    
    # Inverse FFT
    fbm_signal = np.fft.ifft(filtered_F).real
    
    # Take sign to get binary sequence
    signs = np.sign(fbm_signal)
    # Replace zeros with 1
    signs[signs == 0] = 1
    
    return signs


def construct():
    start_time = time.time()
    
    # Initialize with random signs
    np.random.seed(42)
    best_h = np.ones(N)
    best_bound = _compute_c3_bound(best_h)
    
    # Try a few random initializations
    for trial in range(5):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        h = np.random.choice([-1.0, 1.0], size=N)
        bound = _compute_c3_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()
    
    # Try fBm-based initializations with different Hurst exponents
    hurst_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for H in hurst_values:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        h = _generate_fbm_signal(N, H, seed=42 + int(H * 100))
        bound = _compute_c3_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()
    
    # Local search from best initialization
    if time.time() - start_time < TIME_LIMIT - 10:
        best_h, best_bound = _local_search(best_h, max_iter=3000)
    
    # Additional restarts if time permits
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 15:
        for trial in range(3):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            h = np.random.choice([-1.0, 1.0], size=N)
            bound = _compute_c3_bound(h)
            if bound < best_bound:
                h, bound = _local_search(h, max_iter=1000)
                if bound < best_bound:
                    best_bound = bound
                    best_h = h.copy()
    
    # Final check
    final_bound = _compute_c3_bound(best_h)
    if final_bound < best_bound:
        best_bound = final_bound
    
    return best_h


# Test
if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"First 20: {h[:20]}")
    print(f"C3 bound: {_compute_c3_bound(h)}")
    print(f"Score: {1.0 / _compute_c3_bound(h)}")
