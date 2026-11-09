"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: linear + sine harmonics + Gaussian bumps.
    params[0]: linear slope
    params[1]: linear intercept
    params[2]: amplitude for sin(x)
    params[3]: amplitude for sin(2x)
    params[4]: amplitude for sin(3x)
    params[5]: amplitude for cos(x)
    params[6]: amplitude for cos(2x)
    params[7]: amplitude for cos(3x)
    params[8]: amplitude for Gaussian bump at x=0
    params[9]: amplitude for Gaussian bump at x=1
    """
    # Linear part
    y_linear = params[0] * x + params[1]
    
    # Sine harmonics
    y_sin1 = params[2] * np.sin(x)
    y_sin2 = params[3] * np.sin(2 * x)
    y_sin3 = params[4] * np.sin(3 * x)
    
    # Cosine harmonics
    y_cos1 = params[5] * np.cos(x)
    y_cos2 = params[6] * np.cos(2 * x)
    y_cos3 = params[7] * np.cos(3 * x)
    
    # Gaussian bumps (localized features)
    sigma = 1.0
    y_gauss0 = params[8] * np.exp(-0.5 * x**2)
    y_gauss1 = params[9] * np.exp(-0.5 * (x - 1.0)**2)
    
    return y_linear + y_sin1 + y_sin2 + y_sin3 + y_cos1 + y_cos2 + y_cos3 + y_gauss0 + y_gauss1
# EVOLVE-BLOCK-END
