import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod.
    
    Uses a rational function form that captures the elastic-plastic transition
    and temperature dependence.
    
    sigma = N(strain, temp) / D(strain, temp)
    
    N = strain * [ (p0 + p1*T + p2*T^2) + (p3 + p4*T)*strain + (p5 + p6*T)*strain^2 ]
    D = 1 + (p7 + p8*T)*strain + p9*strain^2
    
    Parameters (10):
    p0, p1, p2: Temperature-dependent intercept in numerator (elastic slope region)
    p3, p4: Temperature-dependent linear strain term in numerator
    p5, p6: Temperature-dependent quadratic strain term in numerator
    p7, p8: Temperature-dependent linear strain term in denominator (softening)
    p9: Quadratic strain term in denominator
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale parameters to help BFGS converge from all-ones initialization
    # The strain range is [-0.001911, 0.9636], temp is [0.06667, 1]
    # Scaling by ~10 helps the optimizer see parameter effects
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
    
    # Numerator coefficients with temperature dependence
    a0 = p0 + p1 * T + p2 * T**2
    a1 = p3 + p4 * T
    a2 = p5 + p6 * T
    
    num = u * (a0 + a1 * u + a2 * u**2)
    
    # Denominator: constant term 1 ensures finite slope at u=0
    b1 = p7 + p8 * T
    b2 = p9
    
    den = 1.0 + b1 * u + b2 * u**2
    
    # Safe division
    sigma = num / (den + 1e-12)
    
    return sigma
