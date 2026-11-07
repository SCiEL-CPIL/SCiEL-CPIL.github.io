# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 well-conditioned basis functions defined on [-2, 2].
    
    Basis:
      0: constant (1)
      1: Chebyshev T1 (x)
      2: Chebyshev T2 (2x^2 - 1)
      3: Chebyshev T3 (4x^3 - 3x)
      4: Chebyshev T4 (8x^4 - 8x^2 + 1)
      5: Chebyshev T5 (16x^5 - 20x^3 + 5x)
      6: Chebyshev T6 (32x^6 - 48x^4 + 18x^2 - 1)
      7: Chebyshev T7 (64x^7 - 112x^5 + 56x^3 - 7x)
      8: Chebyshev T8 (128x^8 - 256x^6 + 160x^4 - 32x^2 + 1)
      9: Chebyshev T9 (256x^9 - 576x^7 + 432x^5 - 120x^3 + 9x)
    
    These are orthogonal on [-1, 1] with weight 1/sqrt(1-x^2), and provide
    a well-conditioned design matrix for BFGS fitting on [-2, 2] after scaling.
    """
    # Scale x from [-2, 2] to [-1, 1] for standard Chebyshev polynomials
    t = x / 2.0
    
    # Compute Chebyshev polynomials T0..T9 using recurrence
    # T0 = 1, T1 = t, Tn = 2*t*Tn-1 - Tn-2
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    T5 = 2.0 * t * T4 - T3
    T6 = 2.0 * t * T5 - T4
    T7 = 2.0 * t * T6 - T5
    T8 = 2.0 * t * T7 - T6
    T9 = 2.0 * t * T8 - T7
    
    # Stack basis functions
    basis = np.column_stack([
        T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
    ])
    
    # Linear combination with params
    return basis @ params
# EVOLVE-BLOCK-END
