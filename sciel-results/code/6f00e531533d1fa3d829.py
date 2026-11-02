"""Rewritten: structured optimization for first autocorrelation inequality upper bound."""
import numpy as np
import time

N = 600
# Interval width for [-1/4, 1/4]
h = (1.0 / 2.0) / N

def _autoconvolve_max(f):
    """Compute the maximum of the autoconvolution (normalized) of a step function.
    The autoconvolution is (f * f)(t) = integral f(s) f(t-s) ds.
    For step functions on N equal intervals, this is a discrete convolution scaled by h.
    We normalize by the L1 norm squared? Actually the task says "normalized maximum of its autoconvolution".
    
    The standard first autocorrelation inequality (Hardy-Littlewood) involves:
    C1 = sup { (∫∫ f(x)f(y) K(x-y) dx dy) / (∫ f)^2 } for nonnegative f.
    But here it's specifically about autoconvolution: (f*f)(t).
    
    Let me reconsider: "normalized maximum of its autoconvolution" likely means
    max_t (f*f)(t) / (∫f)^2 or similar. Let's compute (f*f) as discrete convolution * h,
    and normalize by (sum(f)*h)^2 = (∫f)^2.
    
    Actually, for the first inequality, we want to minimize the L-infinity norm of f*f relative to L1 norm squared.
    """
    # f is array of heights, each interval has width h
    # Autoconvolution: c[k] = sum_i f[i] * f[j] where j-i = k, but for full convolution
    # Using FFT for efficiency
    n = 2 * N + 1
    F = np.fft.rfft(f, n=2 * n)
    conv = np.fft.irfft(F * F, n=2 * n)
    # The autoconvolution at position t: for step function, (f*f)(t) = h * sum_{i} f[i] f[j] 
    # where the overlap... Actually discrete convolution of f with f gives:
    # (f discrete * f discrete)[k] = sum_m f[m] f[k-m]
    # Then the continuous autoconvolution at the midpoint of interval k is approximately h * this sum.
    # Let's be more careful: f(x) = f_i for x in [(i-0.5)*h, (i+0.5)*h] ... or [i*h, (i+1)*h]?
    # Intervals on [-1/4, 1/4], so x_i = -1/4 + i*h, i=0..N-1
    # f(x) = f[i] for x in [x_i, x_{i+1})
    # (f*f)(t) = ∫ f(s) f(t-s) ds
    # For t at center of interval k, this is sum over i,j where s in [x_i,x_{i+1}), t-s in [x_j,x_{j+1})
    # This is complex. Let's use the standard approach: 
    # (f*f)(t) for step functions can be computed as: for each pair (i,j), the overlap of [x_i, x_{i+1}) and [t-x_{j+1}, t-x_j)
    # This is O(N^2) which for N=600 is 360,000 ops, very fast.
    
    # Actually, let's use the FFT-based discrete convolution approach which is standard:
    # If we sample f at centers, the autoconvolution at sample points is h * (f * f discrete)[k]
    # where * is circular or linear convolution.
    
    # Let me use direct O(N^2) for accuracy since N=600:
    # f has N intervals. The autoconvolution at the center of interval k (k=0..N-1) is:
    # sum_{i,j} f[i] f[j] * overlap_length([x_i, x_{i+1}), [x_k - x_{j+1}, x_k - x_j])
    # where x_i = -1/4 + i*h
    
    # This is complex. Let me use a simpler, well-known approach:
    # Pad f to length 2N, do linear convolution, multiply by h.
    # The result has 2N-1 entries corresponding to autoconvolution at positions.
    
    padded = np.zeros(2 * N - 1)
    padded[:N] = f
    conv = np.convolve(padded, padded, mode='full')
    # conv has length 2*(2N-1)-1 = 4N-3? No, np.convolve(a, b, 'full') has length len(a)+len(b)-1 = (2N-1)+(2N-1)-1 = 4N-3
    # But the autoconvolution of a function supported on N intervals should be supported on 2N-1 intervals.
    # I think I'm overcomplicating. Let me just use:
    
    # f is the vector of heights. The autoconvolution (f*f) at the center of the k-th interval of the result
    # is h * sum_{i=0}^{N-1} f[i] * f[k-i] for k=0..2N-2, with f out of range being 0.
    # This is the standard discrete convolution scaled by h.
    
    conv_full = np.convolve(f, f, mode='full')
    # conv_full[k] = sum_{m} f[m] f[k-m] for k=0..2N-2
    # The continuous autoconvolution at the center of interval k (of the 2N-1 result intervals) is h * conv_full[k]
    # But wait, the result of convolving two functions supported on [-1/4, 1/4] is supported on [-1/2, 1/2]
    # which has 2N intervals of width h. So there should be 2N-1 or 2N values?
    # Let's just take the max of h * conv_full, and normalize by (sum(f) * h)^2.
    
    max_conv = np.max(conv_full) * h
    l1_norm_sq = (np.sum(f) * h) ** 2
    return max_conv / l1_norm_sq

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Initial guess: triangular (bell-shaped)
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f = np.exp(-8 * x ** 2)
    f = np.clip(f, 0, None)
    best_f = f.copy()
    best_score = _autoconvolve_max(best_f)
    
    # Use Nelder-Mead style local search with random restarts
    import random
    random.seed(42)
    
    n_iter = 0
    while time.time() < deadline:
        n_iter += 1
        # Try a perturbation
        noise_scale = 0.1 / (1 + n_iter * 0.01)
        f_new = best_f + np.random.randn(N) * noise_scale
        f_new = np.clip(f_new, 0, None)
        
        score = _autoconvolve_max(f_new)
        if score < best_score:
            best_f = f_new
            best_score = score
        else:
            # Sometimes accept worse solutions for exploration
            if random.random() < 0.1:
                best_f = f_new
    
    return list(best_f)
