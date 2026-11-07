"""
Static, well-conditioned feature basis for symbolic regression.

Strategy:
The evaluator fits 10 parameters using BFGS starting from ones.
Instead of optimizing the basis structure (which is risky and slow),
we use a fixed, universal basis that spans a wide range of functions.
The basis consists of scaled Chebyshev polynomials (for smooth trends)
and low-frequency Fourier terms (for oscillations).
Features are normalized to have zero mean and unit variance on [-2, 2]
to aid the BFGS optimizer's convergence.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.time()
_DEADLINE = _START + 135.0

# Pre-compute feature normalization statistics on a dense grid
# This ensures the features are well-scaled for the BFGS optimizer
_GRID = np.linspace(-2.0, 2.0, 1000)

def _chebyshev(x, n):
    """Chebyshev polynomial of the first kind."""
    return np.cos(n * np.arccos(np.clip(x, -1.0, 1.0)))

# Define the raw basis functions
# 8 Chebyshev polynomials T_0 to T_7 on x/2 (mapped to [-1, 1])
# 2 Fourier terms: sin(pi*x/2), cos(pi*x/2)
def _raw_basis(x):
    x = np.asarray(x, dtype=float)
    x_norm = x / 2.0  # Map [-2, 2] to [-1, 1]
    
    f = np.empty((len(x), 10), dtype=float)
    
    # Chebyshev polynomials
    for i in range(8):
        f[:, i] = _chebyshev(x_norm, i)
        
    # Fourier terms
    f[:, 8] = np.sin(np.pi * x_norm)
    f[:, 9] = np.cos(np.pi * x_norm)
    
    return f

# Compute mean and std for each feature column on the grid
_F_RAW = _raw_basis(_GRID)
_F_MEAN = _F_RAW.mean(axis=0)
_F_STD = _F_RAW.std(axis=0)
# Avoid division by zero
_F_STD[_F_STD < 1e-6] = 1.0

# Pre-compute the scaling parameters for fast application
# We will center and scale the features in equation()
# F_norm = (F_raw - F_MEAN) / F_STD

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: input array
    params: array of 10 weights
    returns: predictions
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        x = x.ravel()
        
    # Compute raw features
    F = _raw_basis(x)
    
    # Normalize features: (F - mean) / std
    # This makes the BFGS starting point (ones) more reasonable 
    # relative to the scale of the features
    F = (F - _F_MEAN) / _F_STD
    
    # Linear combination
    return F @ np.asarray(params, dtype=float)
# EVOLVE-BLOCK-END
