# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model using a compact polynomial basis
    in (x, v) with explicit cross-terms to capture phase-plane coupling.
    
    Structural change: Instead of RBFs with a shared width parameter, use a
    fixed set of polynomial monomials up to cubic order, including key
    cross-terms (x*v, x^2*v, x*v^2) that are natural for oscillator dynamics.
    This avoids the nonlinearity in the width parameter and provides a smoother
    landscape for BFGS optimization.
    
    Basis functions (10 terms):
    1. 1 (bias)
    2. x
    3. v
    4. x^2
    5. v^2
    6. x*v
    7. x^2*v
    8. x*v^2
    9. x^3
    10. v^3
    
    p0-p9: Coefficients for each basis function
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute all basis functions
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v
    xv2 = x * v2
    
    result = (p0 * np.ones_like(x) +
              p1 * x +
              p2 * v +
              p3 * x2 +
              p4 * v2 +
              p5 * xv +
              p6 * x2v +
              p7 * xv2 +
              p8 * x3 +
              p9 * v3)
    
    return result
# EVOLVE-BLOCK-END
