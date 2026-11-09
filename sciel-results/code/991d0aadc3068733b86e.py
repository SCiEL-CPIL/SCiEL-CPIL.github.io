import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: Piecewise-linear approximation using ramp (clipped linear) functions.
    Each ramp activates at a strain threshold and adds a linear contribution.
    This creates a convex, differentiable surrogate that BFGS can fit efficiently
    while naturally modeling the elastic-yield-hardening regime transitions.
    
    sigma = E(T)*eps + sum_k A_k(T) * max(0, eps - x_k)
    where x_k are fixed strain breakpoints and A_k are fitted amplitudes.
    """
    eps = strain
    T = temp
    
    # Fixed breakpoints for the ramp functions (strain thresholds)
    # These capture: yield point, early hardening, mid hardening, late hardening
    x1 = 0.002  # near-yield
    x2 = 0.01   # early hardening
    x3 = 0.05   # mid hardening  
    x4 = 0.15   # late hardening
    x5 = 0.4    # saturation approach
    
    # Temperature-dependent elastic modulus
    E_T = params[0] * (1.0 + params[1] * T)
    
    # Ramp amplitudes with temperature dependence
    # Each ramp has a base amplitude and a temperature coefficient
    A1 = params[2] + params[3] * T
    A2 = params[4] + params[5] * T
    A3 = params[6] + params[7] * T
    A4 = params[8] + params[9] * T
    
    # Elastic (linear) term
    linear_term = E_T * eps
    
    # Piecewise-linear ramp terms (differentiable, convex)
    ramp1 = A1 * np.maximum(0.0, eps - x1)
    ramp2 = A2 * np.maximum(0.0, eps - x2)
    ramp3 = A3 * np.maximum(0.0, eps - x3)
    ramp4 = A4 * np.maximum(0.0, eps - x4)
    ramp5 = 0.5 * A4 * np.maximum(0.0, eps - x5)  # reuse A4 for 5th ramp with scaling
    
    stress = linear_term + ramp1 + ramp2 + ramp3 + ramp4 + ramp5
    
    return stress
