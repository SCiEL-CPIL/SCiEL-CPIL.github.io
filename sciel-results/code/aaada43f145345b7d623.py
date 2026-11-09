import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model:
    sigma = (E0 + E1*T)*epsilon + (S0 + S1*T)*tanh(K0*epsilon + K1*T) 
            + (Q0 + Q1*T)*epsilon^2 + T2*T^2 + C
    
    Parameters:
    p0: E0, p1: E1
    p2: S0, p3: S1
    p4: K0, p5: K1
    p6: Q0, p7: Q1
    p8: T2
    p9: C
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Elastic term
    elastic = (p0 + p1 * temp) * strain
    
    # Hardening term (smooth saturation)
    # Clip the argument to tanh to prevent numerical issues, though tanh is stable
    # The argument is p4*strain + p5*temp
    arg = p4 * strain + p5 * temp
    # tanh is bounded between -1 and 1. 
    # We want a term that starts at 0 and grows. 
    # tanh(x) is approx x for small x. 
    # This adds to the linear term effectively increasing slope initially.
    hardening = (p2 + p3 * temp) * np.tanh(arg)
    
    # Quadratic correction
    quadratic = (p6 + p7 * temp) * strain**2
    
    # Temperature non-linearity
    temp_quad = p8 * temp**2
    
    # Constant
    const = p9
    
    return elastic + hardening + quadratic + temp_quad + const
