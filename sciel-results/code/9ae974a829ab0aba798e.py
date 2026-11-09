import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: Elastic + yield saturation + plastic hardening via dual-tanh.
    sigma = E(T)*eps + A(T)*tanh(B(T)*eps) + C(T)*tanh(D(T)*eps)
    
    The dual-tanh structure captures both the yield point (first tanh)
    and the subsequent hardening saturation (second tanh). Temperature
    dependence is linear in T for all coefficients.
    """
    eps = strain
    T = temp
    
    E0 = params[0]
    c_E = params[1]
    A0 = params[2]
    A1 = params[3]
    B0 = params[4]
    B1 = params[5]
    C0 = params[6]
    C1 = params[7]
    D0 = params[8]
    D1 = params[9]
    
    # Temperature-dependent coefficients
    E_T = E0 * (1.0 + c_E * T)
    A_T = A0 + A1 * T
    B_T = B0 + B1 * T
    C_T = C0 + C1 * T
    D_T = D0 + D1 * T
    
    # Linear elastic term
    linear_term = E_T * eps
    
    # Yield saturation term (first tanh)
    tanh1_arg = B_T * eps
    tanh1_arg = np.clip(tanh1_arg, -50, 50)
    tanh_term1 = A_T * np.tanh(tanh1_arg)
    
    # Plastic hardening saturation term (second tanh)
    tanh2_arg = D_T * eps
    tanh2_arg = np.clip(tanh2_arg, -50, 50)
    tanh_term2 = C_T * np.tanh(tanh2_arg)
    
    stress = linear_term + tanh_term1 + tanh_term2
    
    return stress
