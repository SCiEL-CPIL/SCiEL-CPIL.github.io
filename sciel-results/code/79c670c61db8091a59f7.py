"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Learnable Fourier Basis Expansion.
    
    Strategy:
    Instead of fixed integer frequencies (standard Fourier series) or polynomial
    degrees, we use 4 terms with free frequencies.
    
    Parameters (10 total):
    p0: amplitude for term 1
    p1: frequency for term 1 (x-component)
    p2: frequency for term 1 (y-component)
    p3: amplitude for term 2
    p4: frequency for term 2 (x-component)
    p5: frequency for term 2 (y-component)
    p6: amplitude for term 3
    p7: frequency for term 3 (x-component)
    p8: frequency for term 3 (y-component)
    p9: bias (constant offset)
    
    Model:
    f(x, y) = p9 + p0*sin(p1*x + p2*y) + p3*sin(p4*x + p5*y) + p6*sin(p7*x + p8*y)
    
    Why this helps:
    1. Adaptivity: If the target function has a specific periodicity (e.g., sin(2.5x)),
       the fixed basis would struggle or require high-order harmonics. The learnable
       frequency allows the model to snap to the correct frequency.
    2. Efficiency: 3 sinusoidal terms with free parameters can approximate a wide
       range of smooth functions (including non-periodic ones over [-1,1] as partial
       waves) more flexibly than a fixed low-degree polynomial.
    3. BFGS Compatibility: While non-linear in parameters, the landscape is often
       smoother for sinusoids than for high-degree polynomials or RBFs with sharp
       peaks, provided initialization is reasonable. Starting from all-ones means
       frequencies start at 1.0, which is a valid non-degenerate state.
    """
    if params is None or len(params) < 10:
        return np.zeros_like(x)
    
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    
    # Extract parameters
    w = np.asarray(params, dtype=float)[:10]
    
    # Safety check for non-finite parameters
    if not np.all(np.isfinite(w)):
        return np.zeros_like(x)
    
    # Unpack
    a1, f1x, f1y = w[0], w[1], w[2]
    a2, f2x, f2y = w[3], w[4], w[5]
    a3, f3x, f3y = w[6], w[7], w[8]
    bias = w[9]
    
    # Compute basis terms
    # Using sin for simplicity. Cosine is just a phase-shifted sine, and since
    # frequency and amplitude are free, the combination covers the general form.
    # Note: We could also add cos terms, but we only have 10 params.
    # 3 sin terms + 1 bias = 10 params.
    
    term1 = a1 * np.sin(f1x * x + f1y * y)
    term2 = a2 * np.sin(f2x * x + f2y * y)
    term3 = a3 * np.sin(f3x * x + f3y * y)
    
    result = bias + term1 + term2 + term3
    
    # Ensure output is finite
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
# EVOLVE-BLOCK-END
