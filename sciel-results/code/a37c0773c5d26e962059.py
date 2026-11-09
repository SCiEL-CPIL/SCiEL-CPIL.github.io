"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a hybrid model:
    - Global linear trend: b0 + bx*x + by*y
    - Localized RBF kernels: 5 fixed centers with shared anisotropic width.
    
    Parameters (10 total):
    0-4: Weights for 5 RBFs
    5: Log-width for x (sigma_x = exp(p5))
    6: Log-width for y (sigma_y = exp(p6))
    7-9: Linear coefficients (b0, bx, by)
    """
    # Extract parameters
    w1, w2, w3, w4, w5 = params[0], params[1], params[2], params[3], params[4]
    log_sig_x = params[5]
    log_sig_y = params[6]
    b0, bx, by = params[7], params[8], params[9]
    
    # Compute widths (ensure positivity and numerical stability)
    # Clip to prevent overflow in exp or underflow in division
    # exp(10) is ~22000, exp(-10) is ~0.000045. 
    # If sigma is too small, RBFs become spikes. If too large, they become flat.
    # Clipping log_sigma to [-10, 10] keeps sigma in [4.5e-5, 22000].
    log_sig_x = np.clip(log_sig_x, -10, 10)
    log_sig_y = np.clip(log_sig_y, -10, 10)
    
    sig_x = np.exp(log_sig_x)
    sig_y = np.exp(log_sig_y)
    
    # Pre-compute inverse variances for efficiency
    inv_var_x = 1.0 / (sig_x * sig_x)
    inv_var_y = 1.0 / (sig_y * sig_y)
    
    # Fixed centers
    # c1: (0, 0)
    # c2: (0.5, 0)
    # c3: (-0.5, 0)
    # c4: (0, 0.5)
    # c5: (0, -0.5)
    
    # Compute RBF terms
    # K(x,y) = exp(-(x-cx)^2 * inv_var_x - (y-cy)^2 * inv_var_y)
    
    # Center 1: (0,0)
    dx1 = x
    dy1 = y
    k1 = np.exp(-(dx1 * dx1) * inv_var_x - (dy1 * dy1) * inv_var_y)
    
    # Center 2: (0.5, 0)
    dx2 = x - 0.5
    dy2 = y
    k2 = np.exp(-(dx2 * dx2) * inv_var_x - (dy2 * dy2) * inv_var_y)
    
    # Center 3: (-0.5, 0)
    dx3 = x + 0.5
    dy3 = y
    k3 = np.exp(-(dx3 * dx3) * inv_var_x - (dy3 * dy3) * inv_var_y)
    
    # Center 4: (0, 0.5)
    dx4 = x
    dy4 = y - 0.5
    k4 = np.exp(-(dx4 * dx4) * inv_var_x - (dy4 * dy4) * inv_var_y)
    
    # Center 5: (0, -0.5)
    dx5 = x
    dy5 = y + 0.5
    k5 = np.exp(-(dx5 * dx5) * inv_var_x - (dy5 * dy5) * inv_var_y)
    
    # Combine RBFs
    rbf_part = w1 * k1 + w2 * k2 + w3 * k3 + w4 * k4 + w5 * k5
    
    # Linear part
    linear_part = b0 + bx * x + by * y
    
    pred = linear_part + rbf_part
    
    return pred
# EVOLVE-BLOCK-END
