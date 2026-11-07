import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model combining Gaussian response curves for temperature
    and pH with a Monod substrate term and density inhibition.
    
    Uses smooth parameterization (exp for positive constraints, sigmoid for
    bounded params) to provide a well-conditioned landscape for BFGS optimization.
    
    Parameters (10 total):
    p0: Max growth rate scale (unconstrained)
    p1: log(Ks) - substrate half-saturation
    p2: Substrate scaling factor
    p3: Temperature optimum
    p4: log(sigma_T) - temperature width
    p5: pH optimum
    p6: log(sigma_pH) - pH width
    p7: log(Kb) - density inhibition constant
    p8: logit(inhibition_strength) - mapped to (0,1) via sigmoid
    p9: Bias term
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Smooth positive mappings to ensure well-conditioned optimization
    Ks = np.exp(p1) + 0.01
    sigma_T = np.exp(p4) + 0.1
    sigma_pH = np.exp(p6) + 0.1
    Kb = np.exp(p7) + 0.01
    # Sigmoid for inhibition strength in (0, 1)
    inhib_strength = 1.0 / (1.0 + np.exp(-p8))
    
    # Substrate term: Monod
    S_term = (s / (Ks + s)) * p2
    
    # Temperature term: Gaussian response
    T_term = np.exp(-((temp - p3) ** 2) / (2.0 * sigma_T ** 2))
    
    # pH term: Gaussian response
    pH_term = np.exp(-((pH - p5) ** 2) / (2.0 * sigma_pH ** 2))
    
    # Density inhibition term
    B_term = 1.0 - inhib_strength * (b / (Kb + b))
    
    # Combine all terms
    growth = p0 * S_term * T_term * pH_term * B_term + p9
    
    return growth
