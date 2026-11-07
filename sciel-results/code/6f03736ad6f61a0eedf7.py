"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod using a rational basis function approach.
    
    Uses a combination of polynomial and rational terms to capture the nonlinear
    stress-strain behavior with temperature dependence. This structural change
    moves away from the Ramberg-Osgood formulation to a more flexible basis.
    """
    s = strain
    t = temp
    p = params
    
    # Normalize inputs to improve conditioning
    s_norm = s  # strain is already in [-0.002, 0.96]
    t_norm = (t - 0.06667) / (1.0 - 0.06667)  # normalize temp to [0, 1]
    
    # Build basis functions - a mix of polynomial and rational terms
    # These are designed to be smooth and capture various aspects of the curve
    
    # Linear elastic term
    term1 = s * (p[0] + p[1] * t_norm)
    
    # Quadratic strain term
    term2 = s * s * (p[2] + p[3] * t_norm)
    
    # Cubic strain term
    term3 = s * s * s * (p[4] + p[5] * t_norm)
    
    # Rational term to capture saturation behavior
    denom = 1.0 + p[6] * np.abs(s)
    term4 = p[7] * s / denom
    
    # Temperature-dependent offset
    term5 = p[8] * t_norm + p[9] * t_norm * t_norm
    
    sigma = term1 + term2 + term3 + term4 + term5
    
    return sigma
# EVOLVE-BLOCK-END
