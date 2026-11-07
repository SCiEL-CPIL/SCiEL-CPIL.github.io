"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function f(x, y) using a linear combination of Chebyshev polynomials.
    The basis is chosen to be orthogonal over [-1, 1] x [-1, 1] to aid BFGS convergence.
    
    Basis functions (10 total):
    1. 1
    2. T1(x) = x
    3. T2(x) = 2x^2 - 1
    4. T1(y) = y
    5. T2(y) = 2y^2 - 1
    6. T1(x)T1(y) = xy
    7. T1(x)T2(y) = x(2y^2 - 1)
    8. T2(x)T1(y) = (2x^2 - 1)y
    9. T2(x)T2(y) = (2x^2 - 1)(2y^2 - 1)
    10. T3(x) = 4x^3 - 3x
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Precompute Chebyshev polynomials
    T0 = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    
    T1_y = y
    T2_y = 2 * y**2 - 1
    
    # Construct the 10 basis terms
    # Note: We multiply by params in the sum for efficiency, but here we just build the terms
    # The evaluator will pass params of length 10.
    
    # Term 0: Constant
    term0 = T0
    # Term 1: x
    term1 = T1_x
    # Term 2: T2(x)
    term2 = T2_x
    # Term 3: y
    term3 = T1_y
    # Term 4: T2(y)
    term4 = T2_y
    # Term 5: xy
    term5 = T1_x * T1_y
    # Term 6: x * T2(y)
    term6 = T1_x * T2_y
    # Term 7: T2(x) * y
    term7 = T2_x * T1_y
    # Term 8: T2(x) * T2(y)
    term8 = T2_x * T2_y
    # Term 9: T3(x)
    term9 = T3_x
    
    # Combine with parameters
    pred = (params[0] * term0 +
            params[1] * term1 +
            params[2] * term2 +
            params[3] * term3 +
            params[4] * term4 +
            params[5] * term5 +
            params[6] * term6 +
            params[7] * term7 +
            params[8] * term8 +
            params[9] * term9)
            
    return pred
# EVOLVE-BLOCK-END
