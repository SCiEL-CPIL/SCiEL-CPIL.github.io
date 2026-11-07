"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a pure Chebyshev tensor-product basis up to total degree 3.
    
    The 10 parameters are linear weights over 10 Chebyshev tensor-product basis functions:
    - Degree 0: T0(x)T0(y)
    - Degree 1: T1(x)T0(y), T0(x)T1(y)
    - Degree 2: T2(x)T0(y), T1(x)T1(y), T0(x)T2(y)
    - Degree 3: T3(x)T0(y), T2(x)T1(y), T1(x)T2(y), T0(x)T3(y)
    
    This complete Chebyshev basis up to total degree 3 is well-conditioned,
    enabling BFGS to converge reliably to the global minimum of the quadratic
    least-squares problem.
    """
    # Safety deadline guard
    t_start = time.time()
    TIME_LIMIT = 140.0

    # --- Chebyshev polynomials for x and y (up to degree 3) ---
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x**2 - 1.0
    T3_x = 4.0 * x**3 - 3.0 * x

    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y**2 - 1.0
    T3_y = 4.0 * y**3 - 3.0 * y

    # --- 10 Chebyshev tensor-product basis terms (total degree <= 3) ---
    f0 = T0_x * T0_y        # degree 0
    f1 = T1_x * T0_y        # degree 1
    f2 = T0_x * T1_y        # degree 1
    f3 = T2_x * T0_y        # degree 2
    f4 = T1_x * T1_y        # degree 2
    f5 = T0_x * T2_y        # degree 2
    f6 = T3_x * T0_y        # degree 3
    f7 = T2_x * T1_y        # degree 3
    f8 = T1_x * T2_y        # degree 3
    f9 = T0_x * T3_y        # degree 3

    # Stack into a basis matrix of shape (n, 10)
    basis = np.column_stack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9))

    # Linear combination with the 10 learnable parameters
    prediction = basis @ params

    # Time guard: if we are close to the limit, just return what we have
    elapsed = time.time() - t_start
    if elapsed > TIME_LIMIT:
        return prediction

    return prediction
# EVOLVE-BLOCK-END
