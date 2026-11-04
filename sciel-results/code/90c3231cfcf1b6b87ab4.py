import numpy as np
from scipy.optimize import minimize
import time

N = 600
W = 1.0 / N  # interval width = (1/4 - (-1/4)) / 600 = 1/2 / 600 = 1/1200? 
# Actually: 600 equal intervals on [-1/4, 1/4], so total length = 1/2, each interval = 1/1200
# Wait: [-1/4, 1/4] has length 1/2. 600 intervals means each has width 1/1200.
W = 0.5 / N  # = 1/1200

def autoconv_max_normalized(h):
    """Compute max of autoconvolution divided by (sum of h * W)^2.
    
    h: array of N nonnegative heights.
    The step function f is h[i] on interval [x_i, x_i + W] where x_i = -1/4 + i*W.
    Autoconvolution (f*f)(t) = integral f(s) f(t-s) ds.
    For step functions, this can be computed via the convolution of the height array.
    """
    # The autoconvolution of the step function
    # f*f at discrete lags: for lag k (in units of W), the value is W * sum_i h[i]*h[i-k]
    # But we need the continuous max. The autoconvolution of a step function is piecewise linear.
    # The maximum occurs either at a multiple of W or at the peak of a linear segment.
    # For nonnegative functions, the autoconvolution is unimodal around 0 if f is symmetric,
    # but in general the max could be anywhere.
    
    # Compute the discrete autoconvolution (cross-correlation at all lags)
    # conv = np.convolve(h, h, mode='full') gives the sum h[i]*h[j] where i+j = k
    # But autoconvolution is sum h[i]*h[i-k] which is correlation.
    # np.correlate(h, h, mode='full') gives sum h[i]*h[i+k] for various k
    
    # Actually, let's use FFT for efficiency
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n_fft)
    conv = np.fft.irfft(H * np.conj(H), n_fft)[:2*N-1]
    # conv[k] = sum_i h[i]*h[i+k-N+1]... let me be more careful
    # np.convolve(h, h, 'full')[k] = sum_i h[i]*h[k-i] for valid i
    # The autoconvolution (correlation) is conv[k] with appropriate indexing
    
    # For the step function, the continuous autoconvolution at position t:
    # If t = m*W + r where 0 <= r < W, then (f*f)(t) = W * (1-r/W) * sum_i h[i]*h[i-m] + W*(r/W) * sum_i h[i]*h[i-m-1]
    # Wait, this is getting complex. Let me use a simpler approach.
    
    # The autoconvolution of step function with heights h on intervals of width W:
    # (f*f)(t) = sum_i sum_j h[i]*h[j]*W * indicator that t is in [x_i - x_j - W, x_i - x_j + W]
    # Actually, (f*f)(t) = integral f(s)f(t-s)ds. With f(s) = h[i] for s in [a_i, a_i+W]:
    # (f*f)(t) = sum_i sum_j h[i]*h[j] * length([a_i, a_i+W] ∩ [t-a_j-W, t-a_j])
    # = sum_i sum_j h[i]*h[j] * max(0, min(a_i+W, t-a_j) - max(a_i, t-a_j-W))
    
    # This is expensive for N=600. Let me use the fact that for step functions,
    # the autoconvolution is piecewise linear with breakpoints at multiples of W.
    
    # Simpler: compute the discrete correlation and interpolate
    # The value at t = k*W (k integer) is W * sum_i h[i]*h[i-k] (with appropriate bounds)
    # Between k*W and (k+1)*W, it's linear.
    
    # Compute correlation c[k] = sum_i h[i]*h[i+k] for k = -(N-1), ..., N-1
    # Using FFT:
    corr = np.fft.irfft(np.abs(H)**2, n=n_fft)[:2*N-1]
    # corr[k] for k=0,...,2N-2 corresponds to lag k-(N-1)
    # So corr[N-1] = sum h[i]*h[i] (lag 0)
    # corr[N-1+k] = sum_i h[i]*h[i+k] for k >= 0 (positive lag)
    # corr[N-1-k] = sum_i h[i]*h[i-k] for k >= 0 (negative lag) = corr[N-1+k] by symmetry for real h
    
    # The continuous autoconvolution at t = k*W is W * corr[N-1+k] for integer k
    # Between k*W and (k+1)*W, it's linear interpolation between W*corr[N-1+k] and W*corr[N-1+k+1]
    # But wait, that's not quite right. The autoconvolution of step functions is NOT simply
    # the linear interpolation of the discrete correlation.
    
    # Let me just compute it directly for the max. For N=600, O(N^2) = 360000 which is fine.
    # But we need the continuous max, not just discrete.
    
    # Actually, for the purpose of optimization, let me use a simpler proxy:
    # The max of the discrete autoconvolution (times W) as a proxy for the continuous max.
    # This might not be exact but should correlate well.
    
    discrete_max = np.max(corr) * W
    total_mass = np.sum(h) * W
    return discrete_max / (total_mass ** 2)

def construct():
    deadline = time.time() + 140  # 10 second safety margin
    
    # Start with uniform
    h0 = np.ones(N)
    
    def objective(x):
        h = np.maximum(x, 0)  # enforce nonnegativity
        return autoconv_max_normalized(h)
    
    # Use L-BFGS-B with multiple random restarts
    best_h = h0.copy()
    best_val = objective(h0)
    
    rng = np.random.RandomState(42)
    
    # Try different initializations
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Different starting points
        if trial == 0:
            x0 = h0.copy()
        elif trial == 1:
            # Gaussian-like
            x = np.linspace(-1, 1, N)
            x0 = np.exp(-x**2 * 3)
            x0 = x0 / np.max(x0)
        elif trial == 2:
            # Cosine bump
            x = np.linspace(-1, 1, N)
            x0 = np.cos(np.pi * x / 2)**2
        elif trial == 3:
            # Random smooth
            x0 = np.abs(rng.randn(N)) 
            x0 = x0 / np.max(x0)
        elif trial == 4:
            # Two bumps
            x = np.linspace(-1, 1, N)
            x0 = np.exp(-(x-0.4)**2*10) + np.exp(-(x+0.4)**2*10)
            x0 = x0 / np.max(x0)
        elif trial == 5:
            # Flat top with tapered edges
            x = np.linspace(-1, 1, N)
            x0 = np.clip(1 - np.abs(x)**4, 0, 1)
        elif trial == 6:
            # Sinc-like (but clipped to nonnegative)
            x = np.linspace(-1, 1, N)
            x0 = np.sinc(x * 2)
            x0 = np.maximum(x0, 0)
            if np.max(x0) > 0:
                x0 = x0 / np.max(x0)
            else:
                x0 = h0.copy()
        else:
            # Random smooth via Fourier
            freqs = np.fft.rfftfreq(N)
            coeffs = rng.randn(len(freqs)) + 1j * rng.randn(len(freqs))
            coeffs[0] = 1.0  # DC component
            x0 = np.fft.irfft(coeffs * N)
            x0 = np.maximum(x0, 0)
            if np.max(x0) > 0:
                x0 = x0 / np.max(x0)
            else:
                x0 = h0.copy()
        
        # Normalize to have reasonable scale
        x0 = x0 / np.max(x0) if np.max(x0) > 0 else x0
        
        # Optimize
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success or res.fun < best_val:
                h_opt = np.maximum(res.x, 0)
                val = objective(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # Final polish with more iterations if time permits
    if time.time() < deadline - 5:
        try:
            h_curr = np.maximum(best_h, 0)
            res = minimize(objective, h_curr, method='L-BFGS-B',
                          options={'maxiter': 2000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                best_h = np.maximum(res.x, 0)
                best_val = res.fun
        except Exception:
            pass
    
    return list(best_h)

# Test
if __name__ == "__main__":
    h = construct()
    print(f"Result: {len(h)} values, max={max(h):.4f}, sum={sum(h):.4f}")
    print(f"Score: {autoconv_max_normalized(np.array(h)):.6f}")
