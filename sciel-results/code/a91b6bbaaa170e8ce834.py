import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: Additive basis with independent yield and hardening tanh saturation,
    quadratic strain correction, all with linear temperature dependence.
    sigma = E(T)*eps + A(T)*tanh(B_y*eps) + C(T)*tanh(B_h*eps) + D(T)*eps^2
    
    Independent B_y and B_h let BFGS fit the sharp yield plateau and the
    gradual hardening regime separately. Linear-in-T coefficients keep the
    landscape well-conditioned for BFGS from an all-ones start.
    """
    eps = strain
    T = temp
    
    E0 = params[0]
    E1 = params[1]
    A0 = params[2]
    A1 = params[3]
    B_y = params[4]
    C0 = params[5]
    C1 = params[6]
    B_h = params[7]
    D0 = params[8]
    D1 = params[9]
    
    # Temperature-dependent coefficients (linear in T)
    E_T = E0 + E1 * T
    A_T = A0 + A1 * T
    C_T = C0 + C1 * T
    D_T = D0 + D1 * T
    
    # Independent yield and hardening scales
    x_yield = B_y * eps
    x_hard = B_h * eps
    
    linear_term = E_T * eps
    yield_term = A_T * np.tanh(x_yield)
    hard_term = C_T * np.tanh(x_hard)
    quad_term = D_T * eps * eps
    
    stress = linear_term + yield_term + hard_term + quad_term
    
    return stress
