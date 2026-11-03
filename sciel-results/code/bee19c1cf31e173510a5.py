import numpy as np
from scipy import optimize
import time

N = 600
h = 1.0 / 1200  # interval width

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution."""
    heights = np.asarray(heights, dtype=np.float64)
    heights = np.maximum(heights, 0.0)  # enforce nonnegativity
    
    # L1 and L2 norms
    l1 = h * np.sum(heights)
    l2_sq = h * np.sum(heights**2)
    
    if l1 <= 1e-15 or l2_sq <= 1e-15:
        return 1e10
    
    # Autoconvolution via FFT
    # We need to compute (f*f)(t) = integral f(s) f(t-s) ds
    # For step functions, this can be computed efficiently
    # Pad to avoid circular convolution
    n_fft = 1 << (2 * N).bit_length()
    f_pad = np.zeros(n_fft)
    f_pad[:N] = heights * h  # scale by h since f(x) = heights on each interval of width h
    
    # Actually, let's think more carefully
    # f(x) = h_i for x in [i*h, (i+1)*h) for i = 0..N-1, where the interval covers [-1/4, 1/4]
    # So f is defined on [-1/4, 1/4], total width 1/2
    # Autoconvolution (f*f)(t) = integral_{-inf}^{inf} f(s) f(t-s) ds
    # The result is supported on [-1/2, 1/2]
    
    # Using FFT: autoconv = ifft(fft(f) * fft(f))
    f_ext = np.zeros(n_fft)
    f_ext[:N] = heights  # values on each subinterval
    # But we need to account for the width h
    # The actual function value is heights[i] on interval i of width h
    # So in the integral, we have f(s) ds = heights[i] * h for s in interval i
    
    # Let's use a cleaner approach: sample the autoconvolution at grid points
    # (f*f)(t) for t at multiples of h
    # This is a discrete convolution scaled by h
    conv = np.convolve(heights, heights, mode='full') * h
    # conv[k] corresponds to (f*f)(k*h) approximately, for k = 0..2N-2
    
    max_conv = np.max(conv)
    
    # Normalization: I'll try max_conv / (l1**2 / N) or similar
    # For uniform heights=1: l1 = h*N = 1/2, l1^2 = 1/4
    # conv for uniform: conv[k] = h * min(k+1, N, 2N-1-k) for k=0..2N-2
    # max at k=N-1: conv[N-1] = h*N = 1/2
    # So max_conv = 1/2, l1^2 = 1/4, ratio = 2
    # But seed scores 1.0, so normalization must be different
    
    # Try: max_conv / l2_sq
    # l2_sq = h * N * 1 = 1/2
    # max_conv / l2_sq = (1/2) / (1/2) = 1.0  ✓
    
    return max_conv / l2_sq

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin
    
    # Initial guess: uniform
    best_heights = np.ones(N)
    best_score = compute_score(best_heights)
    
    # Try a few structured initial guesses
    # Gaussian-like
    x = np.linspace(-0.5, 0.5, N)
    gauss = np.exp(-x**2 / (2 * 0.1**2))
    score_g = compute_score(gauss)
    if score_g < best_score:
        best_score = score_g
        best_heights = gauss.copy()
    
    # Triangular
    tri = np.maximum(0, 1 - 2*np.abs(x))
    score_t = compute_score(tri)
    if score_t < best_score:
        best_score = score_t
        best_heights = tri.copy()
    
    # Now use gradient-based optimization
    # We need the gradient of the score
    # score = max(conv) / l2_sq where conv = h * (heights * heights) (discrete conv)
    # This is complex. Let's use numerical gradient or a simpler approach.
    
    # Alternative: use scipy minimize with numerical gradient
    # But 600 dimensions with numerical gradient is expensive.
    
    # Let's try a different approach: use the fact that for the optimal solution,
    # the autoconvolution should be as flat as possible.
    
    # Try L-BFGS-B with a subset of iterations
    from scipy.optimize import minimize
    
    def objective(v):
        v = np.maximum(v, 0.0)
        return compute_score(v)
    
    # Use BFGS with limited iterations
    try:
        result = minimize(
            objective,
            best_heights,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        if result.success or result.fun < best_score:
            new_heights = np.maximum(result.x, 0.0)
            new_score = compute_score(new_heights)
            if new_score < best_score:
                best_score = new_score
                best_heights = new_heights
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return list(best_heights)
    
    # Try more optimization rounds if time permits
    # Use a different starting point
    for trial in range(3):
        if time.time() - start_time > deadline - 5:
            break
        
        # Perturb the best solution
        noise = np.random.randn(N) * 0.1 * np.max(best_heights)
        perturbed = np.maximum(best_heights + noise, 0.0)
        
        try:
            result = minimize(
                objective,
                perturbed,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if result.fun < best_score:
                new_heights = np.maximum(result.x, 0.0)
                new_score = compute_score(new_heights)
                if new_score < best_score:
                    best_score = new_score
                    best_heights = new_heights
        except Exception:
            pass
    
    return list(np.maximum(best_heights, 0.0))
