"""Program using low-rank Gaussian mixture basis with differential evolution and local polishing."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import fftconvolve

N = 600

# Precompute grid points for the step function on [-1/4, 1/4]
_dx = (1.0 / 2.0) / N
_centers = -1.0 / 4.0 + (np.arange(N) + 0.5) * _dx

# Fine grid for convolution calculation
N_FINE = 8192
T_FINE = np.linspace(-0.5, 0.5, N_FINE)
DT_FINE = T_FINE[1] - T_FINE[0]

def _gaussian_mixture_heights(params, n_basis):
    """
    params: array of length 3 * n_basis
            [c1, w1, a1, c2, w2, a2, ...]
    Returns: array of N nonnegative step heights.
    """
    heights = np.zeros(N)
    for i in range(n_basis):
        c = params[3 * i]
        w = params[3 * i + 1]
        a = params[3 * i + 2]
        # Gaussian: a * exp(-(x - c)^2 / (2 * w^2))
        heights += a * np.exp(-(( _centers - c) ** 2) / (2.0 * w ** 2))
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
    
    # Evaluate the step function on the fine grid
    idx = np.floor((T_FINE + 0.25) / _dx).astype(int)
    idx = np.clip(idx, 0, N - 1)
    
    in_support = (T_FINE >= -0.25) & (T_FINE < 0.25)
    f_fine = np.zeros(N_FINE)
    f_fine[in_support] = heights[idx[in_support]]
    
    # Compute autoconvolution on the fine grid
    conv_fine = fftconvolve(f_fine, f_fine, mode='full') * DT_FINE
    max_conv = np.max(conv_fine)
    
    return max_conv / l2_sq

def _optimize_for_basis_count(n_basis, seed=42):
    """
    Run Differential Evolution and then Local Minimization for a specific number of basis functions.
    """
    bounds = []
    for _ in range(n_basis):
        bounds.append((-0.25, 0.25))  # center
        bounds.append((0.005, 0.5))   # width
        bounds.append((0.0, 5.0))     # amplitude
    
    def objective(params):
        heights = _gaussian_mixture_heights(params, n_basis)
        return _compute_score(heights)
    
    # 1. Global Optimization with Differential Evolution
    de_result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=50,
        tol=1e-4,
        popsize=8,
        init='sobol',
        polish=False,
        updating='immediate',
        workers=1
    )
    
    best_params = de_result.x
    best_score = de_result.fun
    
    # 2. Local Polishing with L-BFGS-B
    local_result = minimize(
        objective,
        best_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-8}
    )
    
    if local_result.fun < best_score:
        best_params = local_result.x
        best_score = local_result.fun
        
    return best_params, best_score

def construct():
    # Try different numbers of basis functions to find the best structure
    configs = [
        (4, 42),
        (6, 123),
        (8, 456),
        (10, 789)
    ]
    
    best_overall_params = None
    best_overall_score = 1e10
    best_n_basis = 0
    
    for n_basis, seed in configs:
        try:
            params, score = _optimize_for_basis_count(n_basis, seed)
            if score < best_overall_score:
                best_overall_score = score
                best_overall_params = params
                best_n_basis = n_basis
        except Exception as e:
            pass
            
    if best_overall_params is None:
        # Fallback: single Gaussian centered at 0
        return [1.0] * N
        
    best_heights = _gaussian_mixture_heights(best_overall_params, best_n_basis)
    best_heights = np.maximum(best_heights, 0.0)
    
    return list(best_heights)
