"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form bivariate function using a complete degree-3 Chebyshev tensor product basis.
    
    The basis consists of 10 terms:
    T_i(x) * T_j(y) where i + j <= 3 is not strictly enforced, but we select the 10 lowest-order
    tensor products to span the space of all bivariate polynomials of degree <= 3.
    
    Specifically:
    b0: T0(x)T0(y) = 1
    b1: T1(x)T0(y) = x
    b2: T0(x)T1(y) = y
    b3: T2(x)T0(y) = 2x^2 - 1
    b4: T0(x)T2(y) = 2y^2 - 1
    b5: T1(x)T1(y) = xy
    b6: T3(x)T0(y) = 4x^3 - 3x
    b7: T2(x)T1(y) = (2x^2 - 1)y
    b8: T1(x)T2(y) = x(2y^2 - 1)
    b9: T0(x)T3(y) = 4y^3 - 3y
    
    This basis is complete for polynomials of degree 3 and is well-conditioned due to the
    orthogonality of Chebyshev polynomials on [-1, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Compute Chebyshev polynomials T0 to T3 for x and y
    # T0 = 1
    # T1 = t
    # T2 = 2t^2 - 1
    # T3 = 4t^3 - 3t
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x * x - 1.0
    T3_x = 4.0 * x * x * x - 3.0 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y * y - 1.0
    T3_y = 4.0 * y * y * y - 3.0 * y
    
    # Construct the 10 basis functions
    b0 = T0_x * T0_y
    b1 = T1_x * T0_y
    b2 = T0_x * T1_y
    b3 = T2_x * T0_y
    b4 = T0_x * T2_y
    b5 = T1_x * T1_y
    b6 = T3_x * T0_y
    b7 = T2_x * T1_y
    b8 = T1_x * T2_y
    b9 = T0_x * T3_y
    
    # Linear combination
    pred = (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
    
    # Ensure output is finite
    if not np.all(np.isfinite(pred)):
        return np.mean(y) * np.ones_like(x)
        
    return pred
# EVOLVE-BLOCK-END
