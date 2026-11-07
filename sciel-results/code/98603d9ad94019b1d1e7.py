"""
Strategy: Bilinear Chebyshev product basis.
Uses products T_i(x)*T_j(x) for selected pairs to capture sharp features.
Linear in the 10 parameters, so BFGS converges to the global minimum.
"""
import numpy as np
import time

# Pre-compute Chebyshev polynomial coefficients for efficient evaluation
# T_0(x) = 1
# T_1(x) = x
# T_2(x) = 2x^2 - 1
# T_3(x) = 4x^3 - 3x
# T_4(x) = 8x^4 - 8x^2 + 1
# T_5(x) = 16x^5 - 20x^3 + 5x

def _chebyshev(x: np.ndarray, n: int) -> np.ndarray:
    """Compute Chebyshev polynomial T_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    Tn_minus_2 = np.ones_like(x)
    Tn_minus_1 = x.copy()
    for _ in range(2, n + 1):
        Tn = 2.0 * x * Tn_minus_1 - Tn_minus_2
        Tn_minus_2 = Tn_minus_1
        Tn_minus_1 = Tn
    return Tn_minus_1

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Bilinear Chebyshev product basis with 10 coefficients.
    
    Basis functions:
    0: T_0 * T_0 = 1
    1: T_1 * T_1 = x^2
    2: T_2 * T_2 = (2x^2-1)^2
    3: T_1 * T_2 = x(2x^2-1)
    4: T_3 * T_3 = (4x^3-3x)^2
    5: T_1 * T_3 = x(4x^3-3x)
    6: T_2 * T_3 = (2x^2-1)(4x^3-3x)
    7: T_0 * T_4 = 8x^4-8x^2+1
    8: T_1 * T_4 = x(8x^4-8x^2+1)
    9: T_2 * T_4 = (2x^2-1)(8x^4-8x^2+1)
    """
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=float)
    
    # Compute Chebyshev polynomials up to degree 4
    T0 = np.ones_like(x)
    T1 = x.copy()
    T2 = 2.0 * x**2 - 1.0
    T3 = 4.0 * x**3 - 3.0 * x
    T4 = 8.0 * x**4 - 8.0 * x**2 + 1.0
    
    # Form the 10 basis functions as products
    phi0 = T0 * T0  # 1
    phi1 = T1 * T1  # x^2
    phi2 = T2 * T2  # (2x^2-1)^2
    phi3 = T1 * T2  # x(2x^2-1)
    phi4 = T3 * T3  # (4x^3-3x)^2
    phi5 = T1 * T3  # x(4x^3-3x)
    phi6 = T2 * T3  # (2x^2-1)(4x^3-3x)
    phi7 = T0 * T4  # 8x^4-8x^2+1
    phi8 = T1 * T4  # x(8x^4-8x^2+1)
    phi9 = T2 * T4  # (2x^2-1)(8x^4-8x^2+1)
    
    # Linear combination
    y = (params[0] * phi0 +
         params[1] * phi1 +
         params[2] * phi2 +
         params[3] * phi3 +
         params[4] * phi4 +
         params[5] * phi5 +
         params[6] * phi6 +
         params[7] * phi7 +
         params[8] * phi8 +
         params[9] * phi9)
    
    # Check for non-finite values
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
    
    return y
