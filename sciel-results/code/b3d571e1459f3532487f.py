"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using tanh-based nonlinearities for smoother 
    saturation behavior and better conditioning for BFGS optimization.
    
    Structure: a = p1*tanh(x) + p2*tanh(x)^3 + p3*tanh(v) + p4*tanh(v)^3
               + p5*tanh(x)*tanh(v) + p6*tanh(x)^2*tanh(v) + p7*tanh(x)*tanh(v)^2
               + p8*tanh(x)^2*tanh(v)^2 + p9*tanh(x^2)*tanh(v^2) + p10
    
    Inputs:
        x: position (array)
        v: velocity (array)
        params: array of 10 fitted constants
    """
    # Use tanh for bounded, smooth nonlinearities that saturate gracefully
    # This avoids potential numerical issues with high powers and provides
    # better gradient behavior for BFGS optimizer
    
    tanh_x = np.tanh(x)
    tanh_v = np.tanh(v)
    
    # Precompute powers of tanh values
    tanh_x2 = tanh_x * tanh_x
    tanh_x3 = tanh_x2 * tanh_x
    tanh_v2 = tanh_v * tanh_v
    tanh_v3 = tanh_v2 * tanh_v
    
    # Cross terms
    tanh_x_tanh_v = tanh_x * tanh_v
    tanh_x2_tanh_v = tanh_x2 * tanh_v
    tanh_x_tanh_v2 = tanh_x * tanh_v2
    tanh_x2_tanh_v2 = tanh_x2 * tanh_v2
    
    # Higher order saturation terms
    tanh_x2_v2 = np.tanh(x * x) * np.tanh(v * v)
    
    # Construct terms with tanh-based nonlinearities
    term1 = tanh_x
    term2 = tanh_x3
    term3 = tanh_v
    term4 = tanh_v3
    term5 = tanh_x_tanh_v
    term6 = tanh_x2_tanh_v
    term7 = tanh_x_tanh_v2
    term8 = tanh_x2_tanh_v2
    term9 = tanh_x2_v2
    term10 = np.ones_like(x)
    
    return (params[0] * term1 +
            params[1] * term2 +
            params[2] * term3 +
            params[3] * term4 +
            params[4] * term5 +
            params[5] * term6 +
            params[6] * term7 +
            params[7] * term8 +
            params[8] * term9 +
            params[9] * term10)
# EVOLVE-BLOCK-END
