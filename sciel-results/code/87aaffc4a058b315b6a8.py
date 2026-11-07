"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Bilinear Chebyshev basis: T_i(x) * T_j(x) for selected pairs (i,j).
    This allows capturing non-polynomial features like |x| or sharp transitions
    more efficiently than a single linear basis by exploiting multiplicative interactions.
    The model is linear in the 10 parameters, ensuring BFGS converges to the global minimum.
    
    Chebyshev polynomials of the first kind:
    T_0(x) = 1
    T_1(x) = x
    T_2(x) = 2x^2 - 1
    T_3(x) = 4x^3 - 3x
    T_4(x) = 8x^4 - 8x^2 + 1
    T_5(x) = 16x^5 - 20x^3 + 5x
    T_6(x) = 32x^6 - 48x^4 + 18x^2 - 1
    
    We use products T_i(x) * T_j(x) for specific pairs to span a rich function space.
    The 10 basis functions are chosen to cover various degrees and interactions.
    """
    # Compute Chebyshev polynomials up to degree 6
    T0 = np.ones_like(x)
    T1 = x.copy()
    T2 = 2.0 * x**2 - 1.0
    T3 = 4.0 * x**3 - 3.0 * x
    T4 = 8.0 * x**4 - 8.0 * x**2 + 1.0
    T5 = 16.0 * x**5 - 20.0 * x**3 + 5.0 * x
    T6 = 32.0 * x**6 - 48.0 * x**4 + 18.0 * x**2 - 1.0
    
    # Define 10 basis functions as products of Chebyshev polynomials
    # This bilinear structure allows capturing complex features
    b0 = T0 * T0  # = 1
    b1 = T0 * T1  # = x
    b2 = T0 * T2  # = 2x^2 - 1
    b3 = T0 * T3  # = 4x^3 - 3x
    b4 = T0 * T4  # = 8x^4 - 8x^2 + 1
    b5 = T1 * T1  # = x^2
    b6 = T1 * T2  # = x(2x^2 - 1) = 2x^3 - x
    b7 = T1 * T3  # = x(4x^3 - 3x) = 4x^4 - 3x^2
    b8 = T2 * T2  # = (2x^2 - 1)^2 = 4x^4 - 4x^2 + 1
    b9 = T1 * T4  # = x(8x^4 - 8x^2 + 1) = 8x^5 - 8x^3 + x
    
    # Linear combination of basis functions
    y = (params[0] * b0 + params[1] * b1 + params[2] * b2 + params[3] * b3 +
         params[4] * b4 + params[5] * b5 + params[6] * b6 + params[7] * b7 +
         params[8] * b8 + params[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
