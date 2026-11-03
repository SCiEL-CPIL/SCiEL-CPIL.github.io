"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _fast_autoconv(f):
    """Compute full autoconvolution using FFT for speed."""
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    AC = np.fft.irfft(F * F, n=n_fft)
    return AC[:2 * N - 1]

def _objective(f):
    """Objective function: max|autoconv| / (sum f)^2."""
    s = np.sum(f)
    if abs(s) < 1e-10:
        return 1e10
    ac = _fast_autoconv(f)
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)

def construct():
    """Optimize the sequence using Powell's method with multiple restarts."""
    rng = np.random.default_rng(123)
    best_val = np.inf
    best_x = None
    
    # Try a few random initializations to avoid local minima
    for i in range(3):
        # Initialize with uniform random values
        x0 = rng.uniform(-1, 1, N)
        # Ensure the sum is not zero to avoid division by zero issues during optimization
        if abs(np.sum(x0)) < 1e-6:
            x0[0] += 0.1
            
        # Use Powell's method (derivative-free) to handle the non-smooth max function
        res = minimize(_objective, x0, method='Powell', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-9})
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
    # Fallback to ones if optimization fails or returns invalid
    if best_x is None:
        return np.ones(N)
        
    return best_x
# EVOLVE-BLOCK-END
