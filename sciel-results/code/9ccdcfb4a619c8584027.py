import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x from [-3, 3] to [-1, 1]
    u = x / 3.0
    
    # Ensure inputs are finite to avoid NaN/Inf propagation
    u = np.where(np.isfinite(u), u, 0.0)
    
    # Basis functions:
    # We aim for a well-conditioned, diverse basis that can approximate a wide range of smooth functions.
    # 0: Constant
    b0 = np.ones_like(u)
    
    # 1: Linear
    b1 = u
    
    # 2: Quadratic
    b2 = u ** 2
    
    # 3: Cubic
    b3 = u ** 3
    
    # 4: Quartic
    b4 = u ** 4
    
    # 5: Chebyshev T5 (better conditioned than u^5)
    # T5(u) = 16u^5 - 20u^3 + 5u
    b5 = 16.0 * u ** 5 - 20.0 * u ** 3 + 5.0 * u
    
    # 6: Chebyshev T6
    # T6(u) = 32u^6 - 48u^4 + 18u^2 - 1
    b6 = 32.0 * u ** 6 - 48.0 * u ** 4 + 18.0 * u ** 2 - 1.0
    
    # 7: Chebyshev T7
    # T7(u) = 64u^7 - 112u^5 + 56u^3 - 7u
    b7 = 64.0 * u ** 7 - 112.0 * u ** 5 + 56.0 * u ** 3 - 7.0 * u
    
    # 8: Gaussian RBF centered at 0 for local flexibility
    # Using a moderate width to capture non-polynomial shapes
    b8 = np.exp(-2.0 * u ** 2)
    
    # 9: Tanh for saturation/step-like behavior
    # Scaling by 3.0 to make it transition more sharply around 0
    b9 = np.tanh(3.0 * u)
    
    # Combine linearly
    result = (params[0] * b0 +
              params[1] * b1 +
              params[2] * b2 +
              params[3] * b3 +
              params[4] * b4 +
              params[5] * b5 +
              params[6] * b6 +
              params[7] * b7 +
              params[8] * b8 +
              params[9] * b9)
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
