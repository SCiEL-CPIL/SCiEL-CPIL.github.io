# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed scaling constants to map inputs to roughly [-1, 1]
_X_SCALE = 1.0 / 0.5728
_V_SCALE = 1.0 / 0.2821

# Estimated RMS of basis functions for normalization
# Using conservative estimates to keep params O(1)
_SCALES = np.array([
    1.0,   # 1
    0.5,   # u
    0.33,  # u^2
    0.25,  # u^3
    0.5,   # w
    0.33,  # w^2
    0.25,  # w^3
    0.25,  # u * w
    0.16,  # u^2 * w
    0.16   # u * w^2
])

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Uses a normalized polynomial basis with physically-motivated terms.
    """
    u = x * _X_SCALE
    w = v * _V_SCALE
    
    # Compute powers
    u2 = u * u
    u3 = u2 * u
    w2 = w * w
    w3 = w2 * w
    
    # Basis functions - physically motivated for nonlinear oscillator:
    # 0: constant (bias)
    # 1: u (linear restoring force)
    # 2: u^2 (quadratic/asymmetric restoring)
    # 3: u^3 (cubic/Duffing nonlinear restoring)
    # 4: w (linear damping)
    # 5: w^2 (quadratic damping)
    # 6: w^3 (cubic damping)
    # 7: u * w (parametric coupling)
    # 8: u^2 * w (nonlinear parametric)
    # 9: u * w^2 (velocity-squared restoring)
    
    b0 = np.ones_like(x)
    b1 = u
    b2 = u2
    b3 = u3
    b4 = w
    b5 = w2
    b6 = w3
    b7 = u * w
    b8 = u2 * w
    b9 = u * w2
    
    basis = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    basis_norm = basis / _SCALES
    
    return basis_norm @ params
# EVOLVE-BLOCK-END
