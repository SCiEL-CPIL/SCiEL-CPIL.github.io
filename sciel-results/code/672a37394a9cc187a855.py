"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    
    # Non-linear parametric model
    # This approach uses parameters inside non-linear functions (powers, exp, sin)
    # allowing the model to fit exponential, power-law, and oscillatory targets directly.
    
    # 1. Power law term: p[0] * x^p[1] * y^p[2]
    # Handles targets like x^2*y, x*y^0.5, etc.
    term1 = p[0] * np.power(x, p[1]) * np.power(y, p[2])
    
    # 2. Exponential term: p[3] * exp(p[4] * z)
    # Handles targets like exp(z), exp(-z), etc.
    term2 = p[3] * np.exp(p[4] * z)
    
    # 3. Oscillatory term: p[5] * sin(p[6] * x + p[7] * y)
    # Handles periodic components.
    term3 = p[5] * np.sin(p[6] * x + p[7] * y)
    
    # 4. Quadratic interaction term: p[8] * (x - y)^2
    # Handles symmetry breaking or curvature dependent on difference.
    term4 = p[8] * (x - y) ** 2
    
    # 5. Constant offset
    term5 = p[9]
    
    return term1 + term2 + term3 + term4 + term5
# EVOLVE-BLOCK-END
