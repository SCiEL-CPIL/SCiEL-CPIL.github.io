"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute basis functions for a richer model:
# 1. Chebyshev polynomials T_0 to T_5 (6 params) - good for global smooth structure
# 2. Two localized Gaussian bumps (4 params: amplitude and center for each) - for local features
# 3. One sine component (2 params: amplitude and frequency) - for periodic features
# Total: 12 parameters? No, we only have 10 params.
# Let's adjust:
# 1. Chebyshev T_0 to T_4 (5 params)
# 2. Two Gaussian bumps (4 params: 2 amp, 2 centers)
# 3. One sine (1 param: amplitude, fixed freq=1? or 2 params?)
# Let's stick to 10 params exactly.
# Strategy:
# p[0:5]: Chebyshev coefficients T0..T4
# p[5]: Sine amplitude (freq=1)
# p[6]: Cosine amplitude (freq=1)
# p[7]: Gaussian 1 amplitude
# p[8]: Gaussian 1 center
# p[9]: Gaussian 2 amplitude (center fixed at -p[8]? or 0?)
# This is getting complicated. Let's use a simpler, more robust non-linear basis.

# Alternative: Use a small set of basis functions that are non-linear in x but linear in params.
# The optimizer fits the params.
# Basis:
# 1. 1 (constant)
# 2. x
# 3. x^2
# 4. x^3
# 5. x^4
# 6. sin(x)
# 7. cos(x)
# 8. exp(-x^2)  (Gaussian at 0)
# 9. exp(-(x-1)^2) (Gaussian at 1)
# 10. exp(-(x+1)^2) (Gaussian at -1)

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns prediction using a mixed basis: Polynomials, Trigonometric, and Gaussians.
    This allows the model to capture global trends, periodicity, and localized bumps.
    """
    start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.flatten()
        
    try:
        # Build the design matrix
        # p[0]: 1
        # p[1]: x
        # p[2]: x^2
        # p[3]: x^3
        # p[4]: x^4
        # p[5]: sin(x)
        # p[6]: cos(x)
        # p[7]: exp(-x^2)
        # p[8]: exp(-(x-1)^2)
        # p[9]: exp(-(x+1)^2)
        
        basis = np.column_stack((
            np.ones_like(x),
            x,
            x**2,
            x**3,
            x**4,
            np.sin(x),
            np.cos(x),
            np.exp(-x**2),
            np.exp(-(x-1.0)**2),
            np.exp(-(x+1.0)**2)
        ))
        
        pred = basis @ params
        
        # Ensure finite values
        if not np.all(np.isfinite(pred)):
             # Fallback to simple polynomial if something goes wrong
             pred = params[0] + params[1]*x + params[2]*x**2 + params[3]*x**3 + params[4]*x**4
             
    except Exception:
        # Fallback to constant if everything fails
        pred = np.full_like(x, params[0])
        
    # Safety margin check
    elapsed = time.time() - start_time
    if elapsed > 110:
         pass
         
    return pred
# EVOLVE-BLOCK-END
