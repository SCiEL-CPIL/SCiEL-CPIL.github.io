"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard: return a safe fallback if we're running out of time
    # (This is primarily a safety net; the function itself is O(n) and fast)
    t_start = time.time()
    
    n_points = len(x)
    if n_points == 0:
        return np.array([])
        
    # Map x from [-3, 3] to t in [-1, 1] for better numerical conditioning
    t = x / 3.0
    
    # We have 10 parameters.
    # Strategy: Combine the structural idea of a linear basis (from Inspiration 1, which scored well)
    # with a refinement using low-order polynomial terms (from the current program's Chebyshev idea, 
    # but simplified to avoid ill-conditioning issues that caused the huge score).
    #
    # The current program's 9th-degree Chebyshev basis likely suffered from numerical instability
    # or overfitting/oscillation when fitted via BFGS on 200 points, leading to a massive score.
    # Inspiration 1 (linear) scored 1.0, suggesting the target might be close to linear or that
    # simple models generalize better here. Inspiration 2 (Fourier) scored 0.88, suggesting
    # non-linear components exist.
    #
    # We will use a hybrid basis:
    # - params[0]: constant
    # - params[1]: linear (t)
    # - params[2]: quadratic (t^2)
    # - params[3]: cubic (t^3)
    # - params[4]: quartic (t^4)
    # - params[5]: quintic (t^5)
    # - params[6]: sin(pi*t)
    # - params[7]: cos(pi*t)
    # - params[8]: sin(2*pi*t)
    # - params[9]: cos(2*pi*t)
    #
    # This provides flexibility for both polynomial and periodic non-linearities,
    # but keeps the basis functions well-behaved and orthogonal-ish on [-1,1].
    # Low-degree polynomials are much more stable than high-degree Chebyshev for BFGS.
    
    basis = np.zeros((10, n_points))
    
    # Polynomial terms (monomial basis, low degree)
    basis[0, :] = 1.0
    basis[1, :] = t
    basis[2, :] = t * t
    basis[3, :] = t * t * t
    basis[4, :] = t * t * t * t
    basis[5, :] = t * t * t * t * t
    
    # Trigonometric terms (Fourier basis, low frequency)
    basis[6, :] = np.sin(np.pi * t)
    basis[7, :] = np.cos(np.pi * t)
    basis[8, :] = np.sin(2.0 * np.pi * t)
    basis[9, :] = np.cos(2.0 * np.pi * t)
    
    # Check time, though this should be very fast
    if time.time() - t_start > 100:
        # Fallback to linear if something is wrong (should not happen)
        return params[0] * t + params[1]
        
    # Prediction is the linear combination of basis functions with weights params
    return np.einsum('i,ij->j', params, basis)
# EVOLVE-BLOCK-END
