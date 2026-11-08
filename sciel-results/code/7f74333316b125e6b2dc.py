"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_c3(h):
    """
    Compute the upper bound on C3 for a signed step function h of length N.
    The autoconvolution upper bound is related to the maximum of the 
    autoconvolution of h. We compute the discrete autoconvolution and take
    its maximum value as a proxy for the upper bound.
    
    Actually, let's think about what the "upper bound on C3" means here.
    For a signed step function, the autoconvolution at lag 0 is sum(h_i^2) = N.
    The third autocorrelation inequality likely involves the relationship between
    autocorrelation values. A common bound involves the maximum of the 
    autoconvolution away from the origin, or some normalized quantity.
    
    Given the task says "improve an autoconvolution upper bound" and "lower is better",
    and the seed returns np.ones(N) which gives a certain score, let's assume the
    score is the maximum absolute value of the autoconvolution (excluding the 
    peak at lag 0) or some similar measure.
    
    Let's use the maximum of the autoconvolution as the metric. For a binary 
    step function, the autoconvolution can be computed via FFT.
    """
    # Compute autoconvolution via FFT
    n = len(h)
    # Pad to avoid circular convolution
    fft_size = 2 * n
    H = np.fft.rfft(h, fft_size)
    autoconv = np.fft.irfft(H * np.conj(H), fft_size)
    # The autoconvolution is symmetric; the peak is at index 0 (or n/2 depending on shift)
    # For linear convolution, the peak is at the center
    # Let's take the maximum absolute value excluding the main peak
    # Actually, the "upper bound" might be the maximum value of the autoconvolution
    # Let's return the maximum of the autoconvolution
    return np.max(np.abs(autoconv))

def _spectral_sign(omega, phi, x):
    """Compute sign(sin(omega * x + phi)) for array x."""
    return np.sign(np.sin(omega * x + phi))

def _construct_from_params(omega, phi):
    """Construct the step function from spectral parameters."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    return _spectral_sign(omega, phi, x)

def _objective(params):
    """Objective function for optimization: returns the C3 upper bound."""
    omega, phi = params
    h = _construct_from_params(omega, phi)
    return _compute_c3(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial grid of parameters
    # omega is the frequency, phi is the phase
    # Try a range of omega values and phi values
    best_score = float('inf')
    best_params = None
    
    # Coarse grid search first
    omega_range = np.linspace(1, 50, 50)
    phi_range = np.linspace(0, 2 * np.pi, 20)
    
    for omega in omega_range:
        if time.time() > deadline:
            break
        for phi in phi_range:
            if time.time() > deadline:
                break
            h = _construct_from_params(omega, phi)
            score = _compute_c3(h)
            if score < best_score:
                best_score = score
                best_params = (omega, phi)
    
    # Local refinement with Nelder-Mead
    if best_params is not None and time.time() < deadline - 5.0:
        try:
            result = minimize(_objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Final construction
    if best_params is not None:
        h = _construct_from_params(best_params[0], best_params[1])
    else:
        h = np.ones(N)
    
    return h
# EVOLVE-BLOCK-END
