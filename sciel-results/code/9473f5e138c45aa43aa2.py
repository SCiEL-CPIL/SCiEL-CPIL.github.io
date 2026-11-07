import numpy as np

def equation(strain, temp, params):
    """
    Symbolic regression for stress-strain of an aluminium rod.
    
    Strategy: High-Order Polynomial with Temperature-Dependent Coefficients.
    
    Instead of a piecewise-linear model which introduces non-smoothness (even if softened),
    we use a single smooth polynomial function.
    
    Model:
    Stress = sum_{i=0}^{4} (a_i + b_i * temp) * strain^i
    
    Parameters (10 total):
    p[0], p[1]: Coefficients for strain^0 (constant term)
    p[2], p[3]: Coefficients for strain^1 (linear/elastic slope)
    p[4], p[5]: Coefficients for strain^2 (quadratic curvature)
    p[6], p[7]: Coefficients for strain^3 (cubic curvature)
    p[8], p[9]: Coefficients for strain^4 (quartic curvature)
    
    This model is C-infinity differentiable, which is ideal for BFGS.
    It can approximate the yield point by having the higher-order terms
    counteract the linear growth, creating a plateau or inflection,
    and then resume growth for hardening.
    """
    s = strain
    t = temp
    p = params
    
    # Unpack parameters
    # c0(t) = p[0] + p[1]*t
    # c1(t) = p[2] + p[3]*t
    # c2(t) = p[4] + p[5]*t
    # c3(t) = p[6] + p[7]*t
    # c4(t) = p[8] + p[9]*t
    
    c0 = p[0] + p[1] * t
    c1 = p[2] + p[3] * t
    c2 = p[4] + p[5] * t
    c3 = p[6] + p[7] * t
    c4 = p[8] + p[9] * t
    
    # Evaluate polynomial using Horner's method for numerical stability and speed
    # Stress = c4*s^4 + c3*s^3 + c2*s^2 + c1*s + c0
    stress = c4
    stress = stress * s + c3
    stress = stress * s + c2
    stress = stress * s + c1
    stress = stress * s + c0
    
    return stress
