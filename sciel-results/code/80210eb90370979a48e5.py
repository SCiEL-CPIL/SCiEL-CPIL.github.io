"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Learnable Radial Basis Function (RBF) Network.
    
    Structure:
    - 3 RBF centers in 2D space.
    - 3 Widths (sigma) for the Gaussian kernels.
    - 3 Output weights for each RBF.
    - 1 Global bias.
    
    Total parameters: 3 (cx) + 3 (cy) + 3 (sigma) + 3 (weights) + 1 (bias) = 13.
    We only have 10 parameters. 
    Let's adjust the architecture to fit 10 parameters:
    - 2 RBF centers (4 params: cx1, cy1, cx2, cy2)
    - 2 Widths (2 params: sigma1, sigma2)
    - 2 Output weights (2 params: w1, w2)
    - 1 Global bias (1 param)
    - 1 Global scale/frequency modifier? Or maybe just 3 centers with fixed symmetric widths?
    
    Let's try a slightly richer architecture that still fits in 10 params:
    - 2 Centers: (cx1, cy1), (cx2, cy2) -> 4 params
    - 2 Widths: sigma1, sigma2 -> 2 params
    - 2 Weights: w1, w2 -> 2 params
    - 1 Bias: b -> 1 param
    - 1 Global frequency/scale factor applied to inputs? Or maybe a 3rd weight for a fixed center at 0,0?
    
    Let's go with:
    p0: cx1
    p1: cy1
    p2: cx2
    p3: cy2
    p4: sigma1
    p5: sigma2
    p6: w1
    p7: w2
    p8: b (bias)
    p9: w0 (weight for a fixed RBF at center (0,0) with fixed width 1.0)
    
    This provides 3 adaptive/fixed RBFs. The fixed one at (0,0) helps capture global trends if the centers move away, while the two adaptive ones capture local features.
    
    Model:
    f(x, y) = w0 * exp(-((x-0)^2 + (y-0)^2) / (2 * 1.0^2)) 
             + w1 * exp(-((x-cx1)^2 + (y-cy1)^2) / (2 * sigma1^2))
             + w2 * exp(-((x-cx2)^2 + (y-cy2)^2) / (2 * sigma2^2))
             + b
    
    To avoid division by zero or extremely small sigmas causing numerical issues, we use softplus or sigmoid to constrain sigma > 0.
    sigma = 0.1 + 0.9 * sigmoid(param) ensures sigma is in [0.1, 1.0].
    """
    if params is None or len(params) < 10:
        return np.zeros_like(x)
    
    # Extract parameters
    cx1, cy1, cx2, cy2, s1_raw, s2_raw, w1, w2, b, w0 = params
    
    # Constrain sigmas to be positive and reasonable
    # Using sigmoid to map to (0, 1), then scaling to (0.1, 1.0)
    s1 = 0.1 + 0.9 * (1.0 / (1.0 + np.exp(-s1_raw)))
    s2 = 0.1 + 0.9 * (1.0 / (1.0 + np.exp(-s2_raw)))
    
    # Compute distances squared
    # Fixed RBF at (0,0) with sigma=1.0
    d0_sq = x*x + y*y
    rbf0 = np.exp(-d0_sq / 2.0)
    
    # Adaptive RBF 1
    dx1 = x - cx1
    dy1 = y - cy1
    d1_sq = dx1*dx1 + dy1*dy1
    rbf1 = np.exp(-d1_sq / (2.0 * s1 * s1))
    
    # Adaptive RBF 2
    dx2 = x - cx2
    dy2 = y - cy2
    d2_sq = dx2*dx2 + dy2*dy2
    rbf2 = np.exp(-d2_sq / (2.0 * s2 * s2))
    
    # Combine
    result = w0 * rbf0 + w1 * rbf1 + w2 * rbf2 + b
    
    return result
# EVOLVE-BLOCK-END
