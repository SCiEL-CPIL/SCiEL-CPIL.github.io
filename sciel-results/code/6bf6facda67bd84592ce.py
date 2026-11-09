import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Parameters:
    # p[0:5]: Weights for RBF centers
    # p[5]: log-width for x (sigma_x = exp(p[5]))
    # p[6]: log-width for y (sigma_y = exp(p[6]))
    # p[7]: Constant bias
    # p[8]: Linear x term
    # p[9]: Linear y term
    
    w = params[0:5]
    log_sx = params[5]
    log_sy = params[6]
    b0 = params[7]
    b1 = params[8]
    b2 = params[9]
    
    # Calculate widths
    # Clip log widths to prevent extreme values that might cause numerical issues
    # exp(-10) ~ 4.5e-5, exp(10) ~ 22000. 
    # If sigma is too small, exp(-d^2/sigma^2) becomes 0 quickly.
    # If sigma is too large, exp(-d^2/sigma^2) becomes 1 everywhere.
    # Clipping to [-5, 5] seems safe.
    log_sx = np.clip(log_sx, -5, 5)
    log_sy = np.clip(log_sy, -5, 5)
    
    sx = np.exp(log_sx)
    sy = np.exp(log_sy)
    
    # Centers
    cx = np.array([0.0, 0.5, -0.5, 0.0, 0.0])
    cy = np.array([0.0, 0.0, 0.0, 0.5, -0.5])
    
    # Compute RBF values
    # dx = x[:, None] - cx[None, :] -> shape (N, 5)
    # dy = y[:, None] - cy[None, :] -> shape (N, 5)
    
    dx = x[:, None] - cx[None, :]
    dy = y[:, None] - cy[None, :]
    
    # Normalize by width
    dx_norm = dx / sx
    dy_norm = dy / sy
    
    # Squared distance
    d2 = dx_norm**2 + dy_norm**2
    
    # RBF: exp(-0.5 * d2) is standard Gaussian, but exp(-d2) is also fine.
    # Using exp(-d2) makes the kernel narrower for same sigma.
    # Let's stick to exp(-0.5 * d2) for standard Gaussian RBF behavior.
    kernels = np.exp(-0.5 * d2)
    
    # Weighted sum of kernels
    # kernels shape (N, 5), w shape (5,)
    rbf_part = kernels @ w
    
    # Linear part
    linear_part = b0 + b1 * x + b2 * y
    
    return rbf_part + linear_part
