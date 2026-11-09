import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Learned RBF Network Strategy.
    
    Instead of fixed basis functions (like polynomials or sines), we use 
    Radial Basis Functions (Gaussians) where the centers and widths are 
    themselves parameters to be optimized. This allows the model to 
    'discover' the relevant features of the hidden function.
    
    Parameters:
    - p[0..3]: Centers of 4 Gaussian RBFs
    - p[4..7]: Widths (inverse sigma) of 4 Gaussian RBFs
    - p[8..9]: Weights for the RBFs (and a constant term)
    
    Note: We use 4 RBFs to leave room for a constant bias term (p[9]) 
    and allow the optimizer to adjust the global scale.
    """
    start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Extract parameters
    centers = p[0:4]
    # Ensure widths are positive and non-zero to avoid division by zero or overflow
    # We use a softplus-like or abs+epsilon approach for stability during BFGS
    # abs(p) + 0.1 ensures width > 0
    widths = np.abs(p[4:8]) + 0.1
    weights = p[8:10]
    
    # Calculate RBF terms
    # y = w0 * G(x; c0, w0) + w1 * G(x; c1, w1) + ... + bias
    # G(x; c, w) = exp(-w * (x - c)^2)
    
    # To handle potential overflow in exp, we clip the exponent
    # Max exponent magnitude should be kept reasonable
    y_hat = np.zeros_like(x)
    
    for i in range(4):
        diff = x - centers[i]
        # Exponent is -widths[i] * diff^2
        # If width is large and diff is large, this can be very negative -> 0
        # If width is small, exponent is close to 0 -> 1
        # We don't expect positive exponents here as width > 0 and diff^2 >= 0
        exponent = -widths[i] * diff**2
        # Clip exponent to [-700, 700] to prevent under/overflow issues in exp
        # Underflow to 0 is fine, overflow is not possible with negative exponent
        # But just in case of numerical weirdness
        exponent = np.clip(exponent, -700.0, 700.0)
        y_hat += weights[i] * np.exp(exponent)
        
        # Time guard
        if time.time() - start_time > 10.0:
             break
             
    # Add the constant bias term (p[9])
    y_hat += p[9]
    
    return y_hat
