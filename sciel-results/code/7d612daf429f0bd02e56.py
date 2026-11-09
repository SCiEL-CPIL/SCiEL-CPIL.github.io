"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y using a hybrid basis:
    - 6 Chebyshev polynomials T_0..T_5 for smooth low-frequency features
    - 4 Radial Basis Functions (Gaussian) centered at fixed locations for local features
    
    params[0:6] are coefficients for Chebyshev polynomials
    params[6:10] are coefficients for Gaussian RBFs
    
    This structural change allows the model to capture both global smooth trends
    and local sharp features/discontinuities that a pure polynomial of fixed degree
    might struggle with.
    """
    # Deadline guard: start time
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin from 120s limit
    
    # Scale x from [-2, 2] to [-1, 1] for Chebyshev
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_5
    T_0 = np.ones_like(t)
    T_1 = t.copy()
    
    y = params[0] * T_0 + params[1] * T_1
    
    T_k_minus_2 = T_0
    T_k_minus_1 = T_1
    
    for k in range(2, 6):
        T_k = 2.0 * t * T_k_minus_1 - T_k_minus_2
        y += params[k] * T_k
        T_k_minus_2 = T_k_minus_1
        T_k_minus_1 = T_k
        
    # Check deadline
    if time.time() > deadline:
        return y
    
    # Compute Gaussian RBFs with fixed centers
    # Centers chosen to spread across the domain [-2, 2]
    centers = np.array([-1.5, -0.5, 0.5, 1.5])
    
    # Widths (sigma) for the Gaussians - fixed at 0.5 for simplicity
    sigma = 0.5
    sigma_sq_2 = 2.0 * sigma * sigma
    
    for i, c in enumerate(centers):
        # params[6+i] is the coefficient for this RBF
        rbf = np.exp(-((x - c) ** 2) / sigma_sq_2)
        y += params[6 + i] * rbf
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return y
# EVOLVE-BLOCK-END
