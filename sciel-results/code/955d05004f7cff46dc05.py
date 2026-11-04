"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.signal import fftconvolve

N = 600
TIME_LIMIT = 110  # seconds, safety margin before 120s kill

# EVOLVE-BLOCK-START

def _score(heights):
    """Compute the normalized maximum of autoconvolution.
    
    The step function f(x) has heights[i] on interval [x_i, x_{i+1}] where
    x_i = -1/4 + i * (1/2)/N for i = 0..N-1.
    
    Autoconvolution: (f*f)(t) = integral f(s) f(t-s) ds
    Normalized max: max_t (f*f)(t) / (integral f)^2
    """
    h = np.asarray(heights, dtype=float)
    if np.any(h < 0):
        return np.inf
    
    # Interval width
    dx = (0.25) / N  # total length is 1/2, so dx = 0.5/N
    
    # Autoconvolution via FFT
    # The autoconvolution of a step function with N intervals:
    # (f*f)(t) where t ranges over [-1/2, 1/2]
    # Using FFT: if we pad h to length 2N, convolve h with h
    # The result has 2N-1 points, but we need to be careful about the grid
    
    # Actually, let's think more carefully.
    # f is a step function with N equal intervals on [-1/4, 1/4].
    # The autoconvolution (f*f)(t) = integral_{-1/4}^{1/4} f(s) f(t-s) ds
    # This is nonzero for t in [-1/2, 1/2].
    
    # Discretize: h[i] is the value on interval i.
    # The autoconvolution can be computed as:
    # For each output bin k (corresponding to t_k = -1/2 + k*dx for k=0..2N-1),
    # (f*f)(t_k) = sum over pairs (i,j) such that the overlap of interval i of f(s)
    # and interval j of f(t-s) contributes.
    
    # Simpler approach: use FFT-based convolution.
    # If we treat h as a discrete signal, the autoconvolution in the discrete sense
    # is conv(h, h). But we need the continuous autoconvolution.
    
    # For a step function with equal intervals, the continuous autoconvolution
    # at the center of each output interval can be approximated by:
    # (f*f)(t_k) ≈ dx * sum_{i} h[i] * h[j] where j corresponds to the overlap.
    
    # Actually, let's use the standard result: if f(x) = sum_i h[i] * 1_{[x_i, x_{i+1}]}(x),
    # then (f*f)(t) is a piecewise linear function. The maximum occurs at some point.
    
    # For simplicity and accuracy, let's compute the autoconvolution on a fine grid
    # using FFT.
    
    # The autoconvolution of two step functions with N intervals each:
    # We can compute it exactly. The result is a piecewise linear function with
    # 2N-1 "knots" (excluding endpoints where it's zero).
    
    # Let's use the formula: (f*f)(t) for t in [x_i + x_j, x_{i+1} + x_j] etc.
    # This is complex. Let's use FFT on a fine grid.
    
    # Alternative: use the fact that for equal intervals, we can compute
    # the autoconvolution coefficients directly.
    
    # The autoconvolution at the midpoint of the k-th output interval (where output
    # has 2N-1 intervals of width dx):
    # Actually, the support of f*f is [-1/2, 1/2], which is 2N intervals of width dx.
    
    # Let me just use scipy's fftconvolve on a fine grid for accuracy.
    
    # Actually, for N=600, we can compute the exact autoconvolution.
    # The autoconvolution of a step function with N equal intervals:
    # Let h = [h_0, h_1, ..., h_{N-1}]
    # The autoconvolution g(t) = (f*f)(t) is piecewise linear.
    # g(t) is nonzero for t in [-1/2, 1/2].
    
    # The value of g at the center of the k-th subinterval of the output:
    # This is getting complex. Let me use a simpler approach.
    
    # Use FFT to compute the autoconvolution on a grid of 2N points.
    # Pad h to length 2N, compute FFT, square, inverse FFT.
    
    n_fft = 2 * N
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h
    
    # Autoconvolution via FFT
    H = np.fft.rfft(h_pad, n=n_fft)
    auto = np.fft.irfft(H * H, n=n_fft) * dx
    
    # The autoconvolution is defined on [-1/2, 1/2] with 2N points
    # The max is the max of auto
    max_auto = np.max(auto)
    
    # Integral of f:
    int_f = np.sum(h) * dx
    
    if int_f <= 0:
        return np.inf
    
    normalized = max_auto / (int_f * int_f)
    return normalized


def construct():
    start_time = time.time()
    
    # Initial guess: uniform
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Try some simple patterns
    # Gaussian-like
    x = np.linspace(-1/4, 1/4, N, endpoint=False) + 0.5/N
    gauss = np.exp(-20 * x**2)
    h_gauss = gauss / np.sum(gauss) * np.sum(np.ones(N))  # normalize integral
    # Actually, let's just try different shapes and normalize
    
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Gaussian centered
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5/N
    for sigma in [0.05, 0.08, 0.1, 0.12, 0.15]:
        g = np.exp(-x**2 / (2 * sigma**2))
        candidates.append(g)
    
    # Cosine bump
    for width in [0.3, 0.5, 0.7, 1.0]:
        c = np.maximum(0, np.cos(np.pi * x / width))
        candidates.append(c)
    
    # Top-hat with rounded edges
    for w in [0.2, 0.3, 0.4, 0.5]:
        t = np.ones(N)
        mask = np.abs(x) < w/2
        t[~mask] = 0
        # Smooth edges
        for i in range(N):
            if not mask[i] and (i > 0 or i < N-1):
                neighbors = []
                if i > 0:
                    neighbors.append(i-1)
                if i < N-1:
                    neighbors.append(i+1)
                if any(mask[j] for j in neighbors):
                    t[i] = 0.5
        candidates.append(t)
    
    # Evaluate candidates
    for cand in candidates:
        cand = np.maximum(0, cand)
        if np.sum(cand) > 0:
            s = _score(cand)
            if s < best_score:
                best_score = s
                best_h = cand.copy()
    
    # Now do local optimization using L-BFGS-B
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 30:
        return list(best_h)
    
    # Use L-BFGS-B to minimize the score
    def objective(h_flat):
        h = np.maximum(0, h_flat)
        return _score(h)
    
    def gradient(h_flat):
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(h_flat)
        for i in range(len(h_flat)):
            h_plus = h_flat.copy()
            h_plus[i] += eps
            h_minus = h_flat.copy()
            h_minus[i] -= eps
            grad[i] = (objective(h_plus) - objective(h_minus)) / (2 * eps)
        return grad
    
    # Use scipy minimize
    result = minimize(
        objective,
        best_h,
        method='L-BFGS-B',
        bounds=[(0, None)] * N,
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if result.fun < best_score:
        best_h = np.maximum(0, result.x)
        best_score = result.fun
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 15:
        return list(best_h)
    
    # Try a few random restarts with local optimization
    np.random.seed(42)
    for restart in range(5):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        # Random smooth perturbation
        h_rand = best_h + 0.1 * np.random.randn(N)
        # Smooth it
        from scipy.ndimage import gaussian_filter1d
        h_rand = gaussian_filter1d(h_rand, sigma=3)
        h_rand = np.maximum(0, h_rand)
        
        result = minimize(
            objective,
            h_rand,
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < best_score:
            best_h = np.maximum(0, result.x)
            best_score = result.fun
    
    return list(best_h)

# EVOLVE-BLOCK-END
