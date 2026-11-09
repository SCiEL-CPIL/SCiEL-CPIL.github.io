import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model:
    sigma = E(T)*eps + S(T)*tanh(k*eps + c) + H(T)*eps*tanh(k*eps + c) + Q*eps^2
    
    Where:
    E(T) = p0 + p1*T
    S(T) = p2 + p3*T
    k    = p4
    c    = p5*T + p6
    H(T) = p7 + p8*T
    Q    = p9
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are float64 arrays
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    
    # Compute components
    elastic_modulus = p0 + p1 * T
    yield_amplitude = p2 + p3 * T
    hardening_slope = p7 + p8 * T
    
    # Transition term
    # p4 is the sharpness, p5*T + p6 is the offset
    transition_arg = p4 * eps + p5 * T + p6
    tanh_term = np.tanh(transition_arg)
    
    # Assemble stress
    stress = (elastic_modulus * eps + 
              yield_amplitude * tanh_term + 
              hardening_slope * eps * tanh_term + 
              p9 * eps * eps)
              
    return stress
