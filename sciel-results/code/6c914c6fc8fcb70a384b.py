"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Normalization constants derived from training ranges
# x: [-0.5728, 0.5275]
# v: [-0.1754, 0.2821]
X_MIN = -0.5728
X_MAX = 0.5275
V_MIN = -0.1754
V_MAX = 0.2821

X_MID = (X_MIN + X_MAX) / 2.0
X_HALF = (X_MAX - X_MIN) / 2.0
V_MID = (V_MIN + V_MAX) / 2.0
V_HALF = (V_MAX - V_MIN) / 2.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form NumPy expression for a nonlinear oscillator.
    Uses a normalized Chebyshev polynomial basis for numerical stability.
    """
    # Normalize inputs to [-1, 1] range to improve conditioning of the basis
    xn = (x - X_MID) / X_HALF
    vn = (v - V_MID) / V_HALF
    
    # Compute Chebyshev polynomials T_0, T_1, T_2, T_3
    # T_0(z) = 1
    # T_1(z) = z
    # T_2(z) = 2z^2 - 1
    # T_3(z) = 4z^3 - 3z
    
    T1_xn = xn
    T2_xn = 2.0 * xn**2 - 1.0
    T3_xn = 4.0 * xn**3 - 3.0 * xn
    
    T1_vn = vn
    T2_vn = 2.0 * vn**2 - 1.0
    T3_vn = 4.0 * vn**3 - 3.0 * vn
    
    # Construct basis matrix (10 columns)
    # 1. Bias
    # 2. Linear x
    # 3. Quadratic x
    # 4. Cubic x
    # 5. Linear v
    # 6. Quadratic v
    # 7. Cubic v
    # 8. x*v interaction
    # 9. x^2*v interaction
    # 10. x*v^2 interaction
    
    # We use np.column_stack to create the design matrix
    # This is efficient and avoids Python loops
    basis = np.column_stack((
        np.ones_like(x),
        T1_xn,
        T2_xn,
        T3_xn,
        T1_vn,
        T2_vn,
        T3_vn,
        T1_xn * T1_vn,
        T2_xn * T1_vn,
        T1_xn * T2_vn
    ))
    
    # Compute prediction
    return basis @ params
# EVOLVE-BLOCK-END
