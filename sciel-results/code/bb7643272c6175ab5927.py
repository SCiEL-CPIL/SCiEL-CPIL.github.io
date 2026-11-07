"""
Hybrid Polynomial-Gaussian Basis for Symbolic Regression.

Strategy:
The previous Chebyshev basis failed, likely due to the target function having
non-smooth features (kinks, steps) or requiring local flexibility that a
global polynomial basis cannot efficiently capture with only 10 coefficients.

This program uses a hybrid basis consisting of:
1. Low-order Polynomials (1, x, x^2, x^3) to capture the global trend.
2. Gaussian Radial Basis Functions (RBFs) centered at key locations to
   capture local variations, discontinuities, or sharp peaks.

The Gaussian centers are spaced across the domain [-2, 2] to ensure
coverage. The widths are chosen to be wide enough to maintain a well-conditioned
design matrix (avoiding near-linear dependency between neighboring Gaussians)
but narrow enough to model local features.

Basis:
F_0 = 1
F_1 = x
F_2 = x^2
F_3 = x^3
F_4 = exp(-(x - 1.5)^2 / 0.5)
F_5 = exp(-(x - 0.75)^2 / 0.5)
F_6 = exp(-(x)^2 / 0.5)
F_7 = exp(-(x + 0.75)^2 / 0.5)
F_8 = exp(-(x + 1.5)^2 / 0.5)
F_9 = exp(-(x)^2 / 2.0)  # Wide global bump

This basis is linear in parameters, making the optimization problem convex
(quadratic MSE), which allows BFGS to converge reliably to the global minimum
for the given basis.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        x = x.ravel()
        
    p = np.asarray(params, dtype=float)
    if p.size != 10:
        p = np.ones(10)
        
    N = len(x)
    if N == 0:
        return np.array([])
        
    F = np.zeros((N, 10), dtype=float)
    
    # Polynomials
    F[:, 0] = 1.0
    F[:, 1] = x
    F[:, 2] = x**2
    F[:, 3] = x**3
    
    # Gaussian RBFs
    # Width sigma = 0.5 -> sigma^2 = 0.25. Using 2*sigma^2 in denominator?
    # Standard Gaussian: exp(-(x-c)^2 / (2*sigma^2)).
    # Let's use sigma=0.5, so denom = 0.5.
    denom = 0.5
    
    centers = [1.5, 0.75, 0.0, -0.75, -1.5]
    
    for i, c in enumerate(centers):
        F[:, 4 + i] = np.exp(-(x - c)**2 / denom)
        
    # Wide Gaussian for global shape
    F[:, 9] = np.exp(-(x**2) / 2.0)
    
    return F @ p
# EVOLVE-BLOCK-END
