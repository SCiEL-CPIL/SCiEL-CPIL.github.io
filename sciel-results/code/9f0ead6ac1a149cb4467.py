"""
Strategy: Tensor-Product Fourier Basis.
Unlike the current Chebyshev-algebraic hybrid, this strategy uses a pure
trigonometric basis: sin(k*pi*x), cos(k*pi*x), sin(m*pi*y), cos(m*pi*y),
and their products sin(k*pi*x)*cos(m*pi*y), etc.
This basis is orthogonal on [-1, 1] (with appropriate normalization, though
BFGS handles scaling), which provides a very smooth and well-conditioned
landscape for gradient-based optimization. It is particularly effective if
the hidden function has any periodic or wave-like components, which are
common in synthetic symbolic regression tasks.

We allocate the 10 parameters to capture:
1. Bias
2. Low-frequency x terms: sin(pi*x), cos(pi*x)
3. Low-frequency y terms: sin(pi*y), cos(pi*y)
4. Interaction terms: sin(pi*x)*sin(pi*y), cos(pi*x)*cos(pi*y), sin(pi*x)*cos(pi*y), cos(pi*x)*sin(pi*y)
5. One higher-frequency term: sin(2*pi*x) to capture potential harmonics.

This creates a global, smooth, and orthogonal representation.
"""
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    # Ensure we have at least 10 parameters
    if len(params) < 10:
        p = np.zeros(10)
        p[:len(params)] = params
    else:
        p = params[:10]
        
    # Compute basis functions
    # Using pi * x and pi * y as the base frequencies since domain is [-1, 1]
    # This maps the domain to [-pi, pi], a standard period for trig functions.
    
    px = np.pi * x
    py = np.pi * y
    
    sin_px = np.sin(px)
    cos_px = np.cos(px)
    sin_py = np.sin(py)
    cos_py = np.cos(py)
    
    # Higher frequency for x
    sin_2px = np.sin(2.0 * px)
    
    # Interaction terms
    sin_px_sin_py = sin_px * sin_py
    cos_px_cos_py = cos_px * cos_py
    sin_px_cos_py = sin_px * cos_py
    cos_px_sin_py = cos_px * sin_py
    
    # Assign parameters
    # p[0]: Bias
    # p[1]: sin(pi*x)
    # p[2]: cos(pi*x)
    # p[3]: sin(pi*y)
    # p[4]: cos(pi*y)
    # p[5]: sin(pi*x)*sin(pi*y)
    # p[6]: cos(pi*x)*cos(pi*y)
    # p[7]: sin(pi*x)*cos(pi*y)
    # p[8]: cos(pi*x)*sin(pi*y)
    # p[9]: sin(2*pi*x)
    
    pred = (p[0] 
            + p[1] * sin_px 
            + p[2] * cos_px 
            + p[3] * sin_py 
            + p[4] * cos_py 
            + p[5] * sin_px_sin_py 
            + p[6] * cos_px_cos_py 
            + p[7] * sin_px_cos_py 
            + p[8] * cos_px_sin_py 
            + p[9] * sin_2px)
    
    # Safety check for time
    if time.time() - start_time > 10:
         # Fallback to simple linear if something goes wrong (unlikely)
         return p[0] + p[1] * x + p[3] * y

    return pred
