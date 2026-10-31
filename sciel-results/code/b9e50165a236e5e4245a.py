"""Program using symmetric low-rank Gaussian mixture basis with differential evolution optimization."""
import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import fftconvolve

N = 600

# Precompute grid points for the step function on [-1/4, 1/4]
_dx = (1.0 / 2.0) / N
_centers = -1.0 / 4.0 + (np.arange(N) + 0.5) * _dx

def _symmetric_gaussian_mixture_heights(params, n_pairs=2):
    """
    params: array of length 2 + 3 * n_pairs
            [w0, a0, c1, w1, a1, c2, w2, a2, ...]
            w0, a0: width and amplitude of the central Gaussian (at 0)
            ci, wi, ai: center, width, amplitude of the i-th pair (at +/- ci)
    Returns: array of N nonnegative step heights.
    """
    n = n_pairs
    heights = np.zeros(N)
    
    # Central Gaussian
    w0 = params[0]
    a0 = params[1]
    heights += a0 * np.exp(-(_centers ** 2) / (2.0 * w0 ** 2))
    
    # Paired Gaussians
    for i in range(n):
        c = params[2 + 3 * i]
        w = params[3 + 3 * i]
        a = params[4 + 3 * i]
        
        # Gaussian at +c
        heights += a * np.exp(-(( _centers - c) ** 2) / (2.0 * w ** 2))
        # Gaussian at -c
        heights += a * np.exp(-(( _centers + c) ** 2) / (2.0 * w ** 2))
        
    return np.maximum(heights, 0.0)

def _compute_score(heights):
    """
    Compute the normalized maximum of the autoconvolution.
    Score = max(f*f) / ||f||_2^2
    """
    # Compute L2 norm squared
    l2_sq = np.sum(heights ** 2) * _dx
    
    if l2_sq <= 1e-15:
        return 1e10  # Avoid division by zero
    
    # Compute autoconvolution on a fine grid
    # Use a moderate grid size to balance accuracy and speed
    n_fine = 8192
    t_fine = np.linspace(-0.5, 0.5, n_fine)
    dt = t_fine[1] - t_fine[0]
    
    # Evaluate the step function on the fine grid
    x_fine = t_fine
    idx = np.floor((x_fine + 0.25) / _dx).astype(int)
    idx = np.clip(idx, 0, N - 1)
    
    in_support = (x_fine >= -0.25) & (x_fine < 0.25)
    f_fine = np.zeros(n_fine)
    f_fine[in_support] = heights[idx[in_support]]
    
    # Compute autoconvolution on the fine grid
    conv_fine = fftconvolve(f_fine, f_fine, mode='full') * dt
    max_conv = np.max(conv_fine)
    
    return max_conv / l2_sq

def construct():
    n_pairs = 2
    # Parameter bounds:
    # w0 in [0.01, 0.5], a0 in [0, 5]
    # ci in [0.01, 0.25], wi in [0.01, 0.5], ai in [0, 5]
    
    bounds = []
    bounds.append((0.01, 0.5))  # w0
    bounds.append((0.0, 5.0))   # a0
    
    for _ in range(n_pairs):
        bounds.append((0.01, 0.25)) # center c (positive)
        bounds.append((0.01, 0.5))  # width w
        bounds.append((0.0, 5.0))   # amplitude a
    
    def objective(params):
        heights = _symmetric_gaussian_mixture_heights(params, n_pairs)
        return _compute_score(heights)
    
    # Use differential evolution with tuned parameters
    # Smaller search space allows for more iterations or larger population
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=150,
        tol=1e-8,
        popsize=15,
        init='sobol',
        polish=True,
        updating='immediate',
        workers=1
    )
    
    best_params = result.x
    best_heights = _symmetric_gaussian_mixture_heights(best_params, n_pairs)
    
    # Ensure nonnegative
    best_heights = np.maximum(best_heights, 0.0)
    
    return list(best_heights)
