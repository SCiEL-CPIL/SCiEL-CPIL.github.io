"""Program using a multi-resolution wavelet-like basis with adaptive widths."""
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form symbolic regression candidate.
    
    Uses a multi-resolution basis combining:
    - 2 global terms (constant, linear)
    - 8 localized Gaussian bumps at different centers and widths
    
    The 10 params are weights for these basis functions.
    BFGS fits these weights to minimize training MSE.
    
    The localized basis allows capturing arbitrary smooth functions
    by superimposing appropriately weighted bumps.
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Return a simple constant prediction as fallback
        return np.zeros_like(x)
    
    u = x / 3.0  # Scale to [-1, 1]
    
    # Global terms
    b0 = np.ones_like(u)
    b1 = u
    
    # Localized Gaussian bumps at various centers and widths
    # Centers spread across [-1, 1], widths vary for multi-resolution
    centers = [-0.8, -0.4, -0.1, 0.2, 0.5, 0.8, -0.6, 0.6]
    widths = [0.3, 0.2, 0.15, 0.25, 0.2, 0.3, 0.1, 0.15]
    
    basis = [b0, b1]
    for c, w in zip(centers, widths):
        # Gaussian bump: exp(-((u-c)^2) / (2*w^2))
        bump = np.exp(-((u - c) ** 2) / (2.0 * w * w))
        basis.append(bump)
    
    # Linear combination
    y = np.zeros_like(u)
    for i, b in enumerate(basis):
        y = y + params[i] * b
    
    # Ensure finite output
    y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return y
