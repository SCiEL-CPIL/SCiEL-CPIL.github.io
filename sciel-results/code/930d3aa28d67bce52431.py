# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 basis functions on [-2, 2] combining:
      - Scaled Chebyshev polynomials T0..T5 (well-conditioned, span polynomials)
      - sin(x), cos(x) (periodic behavior)
      - tanh(x) (saturating nonlinearity)
      - exp(-x^2) (Gaussian bump)
    
    Chebyshev polynomials scaled to [-1,1] via t = x/2 provide a well-conditioned
    design matrix for BFGS. The remaining basis functions add non-polynomial
    expressiveness (periodic, saturating, localized) that pure polynomial bases
    lack.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    # Scale to [-1, 1] for standard Chebyshev
    t = x / 2.0
    
    # Chebyshev polynomials T0..T5 via recurrence: T0=1, T1=t, Tn=2*t*Tn-1 - Tn-2
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    T5 = 2.0 * t * T4 - T3
    
    # Non-polynomial basis functions
    f_sin = np.sin(x)
    f_cos = np.cos(x)
    f_tanh = np.tanh(x)
    f_gauss = np.exp(-x ** 2)
    
    # Stack all 10 basis functions
    basis = np.empty((n, 10), dtype=np.float64)
    basis[:, 0] = T0
    basis[:, 1] = T1
    basis[:, 2] = T2
    basis[:, 3] = T3
    basis[:, 4] = T4
    basis[:, 5] = T5
    basis[:, 6] = f_sin
    basis[:, 7] = f_cos
    basis[:, 8] = f_tanh
    basis[:, 9] = f_gauss
    
    return basis @ params
# EVOLVE-BLOCK-END
