import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 150.0
SAFETY_MARGIN = 5.0
NUM_MODES = 20  # Number of DCT modes to use

# EVOLVE-BLOCK-START
def _fft_convolve(x):
    """Full linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n
    X = np.fft.rfft(x, size)
    C = np.fft.irfft(X * X, size)
    return C[:2 * n - 1]

def compute_score(x):
    """Normalized maximum of autoconvolution: max(autocorr) / autocorr[0]."""
    C = _fft_convolve(x)
    c0 = C[0]
    if c0 <= 0:
        return np.inf
    return float(np.max(C) / c0)

def _build_dct_basis(num_modes):
    """
    Build DCT-II basis matrix for the first num_modes modes.
    Basis[i, j] = sqrt(2/N) * cos(pi * j * (2*i + 1) / (2*N)) for j=0
                  sqrt(4/N) * cos(pi * j * (2*i + 1) / (2*N)) for j>0
    """
    n = N
    basis = np.zeros((n, num_modes))
    for j in range(num_modes):
        if j == 0:
            basis[:, j] = np.sqrt(1.0 / n)
        else:
            basis[:, j] = np.sqrt(2.0 / n) * np.cos(np.pi * j * (2 * np.arange(n) + 1) / (2 * n))
    return basis

# Precompute DCT basis
_DCT_BASIS = _build_dct_basis(NUM_MODES)

def _coeffs_to_heights(coeffs):
    """
    Convert non-negative DCT coefficients to step heights.
    coeffs: shape (NUM_MODES,) non-negative coefficients
    Returns: shape (N,) non-negative heights
    """
    x = _DCT_BASIS @ coeffs
    # Project to non-negative
    x = np.maximum(x, 0.0)
    return x

def _objective_dct(coeffs, smooth_beta=50.0):
    """
    Objective function for DCT coefficient optimization.
    coeffs: shape (NUM_MODES,) non-negative coefficients
    Returns smooth max of autoconvolution / autoconv[0].
    """
    x = _coeffs_to_heights(coeffs)
    C = _fft_convolve(x)
    c0 = C[0]
    if c0 < 1e-10:
        return 1e10
    # Smooth max: (1/beta) * log(sum(exp(beta * C)))
    C_scaled = C / c0
    m = np.max(C_scaled)
    smooth_max = m + np.log(np.sum(np.exp(smooth_beta * (C_scaled - m)))) / smooth_beta
    return smooth_max

def _nelder_mead_optimize(initial_coeffs, time_budget):
    """
    Optimize DCT coefficients using Nelder-Mead.
    """
    t_start = time.time()
    best_coeffs = initial_coeffs.copy()
    best_score = compute_score(_coeffs_to_heights(initial_coeffs))
    
    def obj(coeffs):
        # Enforce non-negativity via soft projection
        coeffs_pos = np.maximum(coeffs, 0.0)
        return _objective_dct(coeffs_pos, smooth_beta=30.0)
    
    try:
        result = minimize(
            obj,
            initial_coeffs,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'xatol': 1e-4,
                'fatol': 1e-6,
                'adaptive': True
            }
        )
        if time.time() - t_start > time_budget:
            return best_coeffs, best_score
        
        if result.success or result.fun < best_score:
            coeffs_opt = np.maximum(result.x, 0.0)
            x_opt = _coeffs_to_heights(coeffs_opt)
            score_opt = compute_score(x_opt)
            if score_opt < best_score:
                best_coeffs = coeffs_opt
                best_score = score_opt
    except Exception:
        pass
    
    return best_coeffs, best_score

def construct():
    t0 = time.time()
    remaining = DEADLINE - SAFETY_MARGIN - (time.time() - t0)
    
    # Start from the seed solution
    eps = (np.sqrt(N) - 1.0) / (N - 1.0)
    x_seed = np.full(N, eps)
    x_seed[0] = 1.0
    
    best_x = x_seed.copy()
    best_score = compute_score(best_x)
    
    # Initialize DCT coefficients
    # Start with all coefficients equal (uniform-like)
    initial_coeffs = np.ones(NUM_MODES)
    initial_coeffs[0] = np.sqrt(N)  # DC component
    
    # Time budget for optimization
    time_budget = max(remaining - 10.0, 5.0)
    
    try:
        best_coeffs, opt_score = _nelder_mead_optimize(initial_coeffs, time_budget)
        if opt_score < best_score:
            best_x = _coeffs_to_heights(best_coeffs)
            best_score = opt_score
    except Exception:
        pass
    
    # Final verification
    if time.time() - t0 < DEADLINE - SAFETY_MARGIN:
        final_score = compute_score(best_x)
        if final_score < best_score:
            best_score = final_score
    
    return best_x.tolist()
# EVOLVE-BLOCK-END
