import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    return float(np.sum(x))

def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _autoconv_values(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]

def _soft_max_abs(conv, beta=1000.0):
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta

def _objective(x, beta=500.0):
    conv = _autoconv_values(x)
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _soft_max_abs(conv, beta) / (s * s)

def _refine(x_init, deadline, beta=500.0):
    # Normalize to unit energy to prevent scaling issues, though metric is scale invariant
    # We optimize x directly.
    def obj(x):
        return _objective(x, beta)
    
    try:
        res = minimize(
            obj,
            x_init,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        if res.success and res.fun < _metric(x_init):
            return res.x, res.fun
        else:
            return x_init, _metric(x_init)
    except Exception:
        return x_init, _metric(x_init)

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    # Candidate 1: Uniform (Theoretical Optimum)
    x_uniform = np.ones(N)
    val_uniform = _metric(x_uniform)
    
    # Candidate 2: Random
    x_rand = rng.normal(0, 1, N)
    val_rand = _metric(x_rand)
    
    # Candidate 3: Barker-like (if we can construct one, or just use a low-peak sequence)
    # For N=400, we can tile a Barker sequence of length 13 or 31?
    # Tiling increases peak sidelobes due to correlation between copies.
    # Better to use a Golay pair construction if possible, or just rely on optimization.
    
    best_x = x_uniform
    best_val = val_uniform
    
    # If uniform is not the best (maybe due to numerical issues or constraints), try refinement
    # But uniform should be the global min.
    # Let's try to refine the uniform sequence to see if we can break the symmetry and improve?
    # No, uniform is the max of S^2 for fixed energy.
    
    # Maybe the "Seed" was not uniform, and the current program is stuck in a local min of a non-convex formulation?
    # The current program uses spectral convexity.
    # Let's try a direct time-domain refinement starting from a slightly perturbed uniform sequence.
    
    x_perturbed = x_uniform + rng.normal(0, 0.01, N)
    x_ref, val_ref = _refine(x_perturbed, deadline, beta=200.0)
    
    if val_ref < best_val:
        best_x = x_ref
        best_val = val_ref
    
    # Try refining the random sequence
    x_rand_ref, val_rand_ref = _refine(x_rand, deadline, beta=200.0)
    if val_rand_ref < best_val:
        best_x = x_rand_ref
        best_val = val_rand_ref
        
    return best_x
# EVOLVE-BLOCK-END
