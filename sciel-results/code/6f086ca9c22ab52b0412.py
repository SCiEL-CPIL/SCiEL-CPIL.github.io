"""Quadratic Polynomial Basis: Global Smooth Basis for Symbolic Regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 135.0  # safety margin before 150s kill

def _check_deadline():
    return time.time() - _start_time > _DEADLINE

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural Change: Quadratic Polynomial Basis.
    Replaces local RBFs with a global quadratic polynomial basis.
    This captures smooth curvature and key interactions (e.g., temperature-substrate)
    while maintaining a convex optimization landscape for BFGS.
    The 10 basis functions are selected to cover bias, linear terms, key quadratics,
    and critical interactions.
    """
    # Normalize inputs to [0, 1] range
    # b: [1, 5] -> [0, 1]
    # s: [1, 40] -> [0, 1]
    # temp: [1, 12] -> [0, 1]
    # pH: [1, 5] -> [0, 1]
    
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 39.0
    t_n = (temp - 1.0) / 11.0
    p_n = (pH - 1.0) / 4.0

    # Construct the 10 basis functions
    # 1. Bias
    # 2. Linear Temp
    # 3. Quadratic Temp (captures optimum)
    # 4. Linear Substrate
    # 5. Quadratic Substrate (captures saturation/inhibition)
    # 6. Linear Inhibitor b
    # 7. Linear pH
    # 8. Interaction Temp * Substrate
    # 9. Interaction Temp * b
    # 10. Interaction Temp * pH
    
    # Vectorized construction for efficiency
    # Shape: (n, 10)
    basis = np.column_stack((
        np.ones_like(b_n),  # 1
        t_n,                 # 2
        t_n * t_n,           # 3
        s_n,                 # 4
        s_n * s_n,           # 5
        b_n,                 # 6
        p_n,                 # 7
        t_n * s_n,           # 8
        t_n * b_n,           # 9
        t_n * p_n            # 10
    ))

    # Linear combination of basis functions
    pred = basis @ params

    return pred

# EVOLVE-BLOCK-END
