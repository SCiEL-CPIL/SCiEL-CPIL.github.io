"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize x to [-1, 1] for better conditioning
    z = x / 3.0
    
    # Define a hybrid basis of 10 functions.
    # This basis is designed to capture a wide variety of smooth functions:
    # Polynomials, trigonometric functions, and non-linear activations.
    
    # 1. Constant
    b0 = np.ones_like(z)
    
    # 2. Linear
    b1 = z
    
    # 3. Quadratic
    b2 = z * z
    
    # 4. Cubic
    b3 = z * z * z
    
    # 5. Sine
    b4 = np.sin(z)
    
    # 6. Cosine
    b5 = np.cos(z)
    
    # 7. Sine (2x)
    b6 = np.sin(2.0 * z)
    
    # 8. Cosine (2x)
    b7 = np.cos(2.0 * z)
    
    # 9. Gaussian bump (centered at 0)
    b8 = np.exp(-z * z)
    
    # 10. Tanh (saturation)
    b9 = np.tanh(z)
    
    # Construct the prediction as a linear combination of the basis functions
    # pred = sum(params[i] * b_i)
    
    pred = (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
            
    return pred
# EVOLVE-BLOCK-END
