"""Gaussian Mixture Model with adaptive local structure."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = None
_TIME_MARGIN = 5.0

def _elapsed_ok():
    if _DEADLINE is None:
        return True
    return (time.time() - _DEADLINE) < _TIME_MARGIN

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    equation(x, y, params) -> predictions
    
    Strategy: 
    Instead of a fixed linear basis, we use a small Gaussian Mixture Model (GMM) 
    structure. The 10 parameters are allocated as follows:
    - params[0]: Global intercept (bias)
    - params[1]: Global linear weight for x
    - params[2]: Global linear weight for y
    - params[3..6]: Center (cx1, cy1, cx2, cy2) of two Gaussian blobs
    - params[7..8]: Widths (sigma1, sigma2) of the two Gaussian blobs (clamped to avoid division by zero)
    - params[9]: Global amplitude scaling for the blobs
    
    This allows the model to fit local features (peaks/valleys) that a global polynomial 
    might struggle with or require high-degree terms to approximate, while the linear 
    part handles the overall trend.
    """
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 150.0

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    
    # Unpack parameters
    # Ensure params is a flat array
    p = np.asarray(params, dtype=float).flatten()
    if p.shape[0] < 10:
        # Fallback if shape is wrong, though contract says length 10
        p = np.pad(p, (0, 10 - p.shape[0]), constant_values=1.0)
    else:
        p = p[:10]
        
    # 1. Global Linear Trend
    bias = p[0]
    wx = p[1]
    wy = p[2]
    
    # 2. Gaussian Blobs
    cx1 = p[3]
    cy1 = p[4]
    cx2 = p[5]
    cy2 = p[6]
    
    # Widths: Use softplus or absolute value + epsilon to ensure positive width
    # Using abs + 0.1 ensures differentiability and positive values
    sigma1 = np.abs(p[7]) + 0.1
    sigma2 = np.abs(p[8]) + 0.1
    
    # Amplitude for blobs
    amp = p[9]
    
    # Compute distances squared
    dx1 = x - cx1
    dy1 = y - cy1
    r2_1 = dx1 * dx1 + dy1 * dy1
    
    dx2 = x - cx2
    dy2 = y - cy2
    r2_2 = dx2 * dx2 + dy2 * dy2
    
    # Gaussian terms: exp(-r^2 / (2 * sigma^2))
    # To avoid underflow/overflow issues in BFGS, we rely on the bounded domain [-1,1]
    # sigma >= 0.1, r^2 <= 8 (max dist in [-1,1]^2 is sqrt(8) ~ 2.8, r^2 ~ 8)
    # Exponent is at most -8 / (2*0.01) = -400, which underflows to 0. This is fine.
    g1 = np.exp(-r2_1 / (2.0 * sigma1 * sigma1))
    g2 = np.exp(-r2_2 / (2.0 * sigma2 * sigma2))
    
    # Combine
    pred = bias + wx * x + wy * y + amp * (g1 + g2)
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return pred
# EVOLVE-BLOCK-END
