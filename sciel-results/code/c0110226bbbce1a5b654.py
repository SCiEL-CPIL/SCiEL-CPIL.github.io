import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model using smooth piecewise-polynomial basis with sigmoid gates.
    
    Structural change: Instead of a fixed tanh hardening term, we use a sum of
    polynomial basis functions (eps^0, eps^1, eps^2, eps^3) where each coefficient
    is modulated by smooth sigmoid gates that depend on strain and temperature.
    This allows the optimizer to shape the elastic-to-plastic transition more flexibly.
    
    Parameters:
    p0: Base linear coefficient (elastic modulus scale)
    p1: Temperature coefficient for linear term
    p2: Gate sharpness for quadratic term activation
    p3: Gate location for quadratic term activation
    p4: Quadratic coefficient scale
    p5: Gate sharpness for cubic term activation  
    p6: Gate location for cubic term activation
    p7: Cubic coefficient scale
    p8: Temperature interaction coefficient
    p9: Intercept
    """
    eps = strain
    T = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure numerical stability: clip sigmoid inputs
    def safe_sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    # Gate 1: activates quadratic term around strain ~p3
    # gate1 ~ 0 for eps << p3, ~ 1 for eps >> p3
    gate1_arg = p2 * (eps - p3)
    gate1 = safe_sigmoid(gate1_arg)
    
    # Gate 2: activates cubic term around strain ~p6
    gate2_arg = p5 * (eps - p6)
    gate2 = safe_sigmoid(gate2_arg)
    
    # Temperature-dependent coefficients
    # Linear term: (p0 + p1*T) * eps
    linear_coeff = p0 + p1 * T
    
    # Quadratic term: p4 * gate1 * (1 + p8*T) * eps^2
    quad_coeff = p4 * gate1 * (1.0 + p8 * T)
    
    # Cubic term: p7 * gate2 * eps^3
    cubic_coeff = p7 * gate2
    
    # Build response
    result = (
        p9
        + linear_coeff * eps
        + quad_coeff * eps * eps
        + cubic_coeff * eps * eps * eps
    )
    
    return result
