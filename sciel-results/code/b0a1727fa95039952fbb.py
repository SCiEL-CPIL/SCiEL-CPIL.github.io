import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod using a rational function
    with quadratic temperature dependence and scaled parameters.
    
    sigma = N(strain, temp) / D(strain, temp)
    
    N = strain * [ (p0 + p1*T + p2*T^2) + (p3 + p4*T)*strain + (p5 + p6*T)*strain^2 ]
    D = 1 + (p7 + p8*T)*strain + p9*strain^2
    
    Parameters (10):
    p0: Elastic modulus base
    p1: Linear temperature effect on modulus
    p2: Quadratic temperature effect on modulus
    p3, p4: Strain-hardening coefficients
    p5, p6: Higher-order strain terms
    p7, p8: Denominator linear terms (transition shape)
    p9: Denominator quadratic term
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale parameters to ensure good gradient magnitudes for BFGS starting from ones
    # The inputs are in [0, 1], so unscaled params near 1 might result in small outputs
    # or flat gradients. Scaling by 10 helps the optimizer see the effect of parameters.
    scale = 10.0
    p0 *= scale
    p1 *= scale
    p2 *= scale
    p3 *= scale
    p4 *= scale
    p5 *= scale
    p6 *= scale
    p7 *= scale
    p8 *= scale
    p9 *= scale
    
    u = strain
    T = temp
    
    # Temperature-dependent coefficients
    # Quadratic in T for the elastic term to capture non-monotonic behavior
    a0 = p0 + p1 * T + p2 * T * T
    a1 = p3 + p4 * T
    a2 = p5 + p6 * T
    
    num = u * (a0 + a1 * u + a2 * u * u)
    
    # Denominator: controls transition and saturation
    b1 = p7 + p8 * T
    b2 = p9
    
    den = 1.0 + b1 * u + b2 * u * u
    
    # Safe division
    sigma = num / (den + 1e-12)
    
    return sigma
