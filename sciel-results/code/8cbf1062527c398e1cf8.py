# EVOLVE-BLOCK-START
import numpy as np

# Fixed scaling constants to map inputs to roughly [-1, 1]
# x range: [-0.5728, 0.5275] -> scale by 1/0.5728
# v range: [-0.1754, 0.2821] -> scale by 1/0.2821
_X_SCALE = 1.0 / 0.5728
_V_SCALE = 1.0 / 0.2821

# Estimated RMS of basis functions to normalize them
# This ensures params are O(1) and BFGS conditioning is good.
# u ~ uniform[-1, 1] approx. RMS(u)=1/sqrt(3)~0.57, RMS(u^2)=1/sqrt(5)~0.44, etc.
# We use simple approximations based on max range to keep it simple and robust.
# Max u ~ 1, Max w ~ 1.
# RMS(u) ~ 0.5, RMS(u^2) ~ 0.33, RMS(u^3) ~ 0.25
# RMS(w) ~ 0.5, RMS(w^2) ~ 0.33, RMS(w^3) ~ 0.25
# RMS(uw) ~ 0.25, RMS(u^2w) ~ 0.16, RMS(uw^2) ~ 0.16

_SCALES = np.array([
    1.0,   # 1
    0.5,   # u
    0.33,  # u^2
    0.25,  # u^3
    0.5,   # w
    0.33,  # w^2
    0.25,  # w^3
    0.25,  # u w
    0.16,  # u^2 w
    0.16   # u w^2
])

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Uses a normalized polynomial basis to improve BFGS convergence.
    """
    u = x * _X_SCALE
    w = v * _V_SCALE
    
    # Compute basis functions
    u2 = u * u
    u3 = u2 * u
    w2 = w * w
    w3 = w2 * w
    
    # Basis vector for each row
    # 0: 1
    # 1: u
    # 2: u^2
    # 3: u^3
    # 4: w
    # 5: w^2
    # 6: w^3
    # 7: u * w
    # 8: u^2 * w
    # 9: u * w^2
    
    # Create a matrix of basis values (N, 10)
    # We can do this efficiently with broadcasting or stacking
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
    
    # Stack into (N, 10)
    basis = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Normalize basis functions by their estimated scales
    # This makes the parameters O(1)
    basis_norm = basis / _SCALES
    
    # Linear combination
    return basis_norm @ params
# EVOLVE-BLOCK-END
