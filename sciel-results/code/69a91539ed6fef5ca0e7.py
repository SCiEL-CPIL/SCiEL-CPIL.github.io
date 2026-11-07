"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid symbolic regression model using diverse non-linear units.
    Combines Tanh, Sin, Gaussian, and Quadratic components with global scale/bias.
    
    Model:
    h1 = tanh(w1*x + b1)
    h2 = sin(w2*x + b2)
    h3 = exp(-(w3*(x - b3))^2)
    h4 = w4*x^2 + b4
    y = S * (h1 + h2 + h3 + h4) + B
    
    Parameters (length 10):
    0-1: w1, b1 (Tanh)
    2-3: w2, b2 (Sin)
    4-5: w3, b3 (Gaussian)
    6-7: w4, b4 (Quadratic)
    8: Global Scale S
    9: Global Bias B
    """
    deadline = time.time() + 100  # Safety margin
    
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure params are finite
    if not np.all(np.isfinite(p)):
        p = np.where(np.isfinite(p), p, 1.0)
        
    # Extract parameters
    w1, b1 = p[0], p[1]
    w2, b2 = p[2], p[3]
    w3, b3 = p[4], p[5]
    w4, b4 = p[6], p[7]
    S = p[8]
    B = p[9]
    
    # Compute components
    # 1. Tanh component
    # Clip argument to avoid extreme values if w1 becomes very large
    arg1 = np.clip(w1 * x + b1, -20, 20)
    h1 = np.tanh(arg1)
    
    # 2. Sin component
    # Sin is bounded, no clipping needed, but large args can lose precision
    # x is small, w2 likely moderate.
    h2 = np.sin(w2 * x + b2)
    
    # 3. Gaussian component
    # exp(-(w3*(x-b3))^2)
    # If w3 is large, the Gaussian becomes very narrow.
    # If w3 is 0, it becomes exp(0)=1.
    z = w3 * (x - b3)
    h3 = np.exp(-z * z)
    
    # 4. Quadratic component
    h4 = w4 * x * x + b4
    
    # Combine
    prediction = S * (h1 + h2 + h3 + h4) + B
    
    # Check for non-finite values
    if not np.all(np.isfinite(prediction)):
        prediction = np.where(np.isfinite(prediction), prediction, 0.0)
        
    return prediction
# EVOLVE-BLOCK-END
