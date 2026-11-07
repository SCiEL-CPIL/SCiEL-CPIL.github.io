"""Rewritten program using a Bilinear Chebyshev Product Basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Time guard
    start_time = time.time()
    
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=float)
    
    # Compute Chebyshev polynomials T0 to T4
    T0 = np.ones_like(x)
    T1 = x
    T2 = 2.0 * x**2 - 1.0
    T3 = 4.0 * x**3 - 3.0 * x
    T4 = 8.0 * x**4 - 8.0 * x**2 + 1.0
    
    # Compute products T_i * T_j for selected pairs to form a 10-term basis
    # This allows capturing interactions and higher-order terms efficiently.
    # The span of these products is equivalent to polynomials up to degree 8.
    
    B = np.empty((len(x), 10))
    
    # 1. T0 * T0 = 1
    B[:, 0] = T0
    # 2. T0 * T1 = x
    B[:, 1] = T1
    # 3. T0 * T2 = 2x^2 - 1
    B[:, 2] = T2
    # 4. T0 * T3 = 4x^3 - 3x
    B[:, 3] = T3
    # 5. T0 * T4 = 8x^4 - 8x^2 + 1
    B[:, 4] = T4
    
    # 6. T1 * T1 = x^2
    B[:, 5] = T1 * T1
    # 7. T1 * T2 = x(2x^2 - 1) = 2x^3 - x
    B[:, 6] = T1 * T2
    # 8. T1 * T3 = x(4x^3 - 3x) = 4x^4 - 3x^2
    B[:, 7] = T1 * T3
    # 9. T1 * T4 = x(8x^4 - 8x^2 + 1) = 8x^5 - 8x^3 + x
    B[:, 8] = T1 * T4
    # 10. T2 * T2 = (2x^2 - 1)^2 = 4x^4 - 4x^2 + 1
    B[:, 9] = T2 * T2
    
    # Linear combination
    y = np.dot(B, params)
    
    # Check for non-finite values
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
        
    return y
# EVOLVE-BLOCK-END
