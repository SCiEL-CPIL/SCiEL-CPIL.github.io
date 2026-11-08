import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress model with rational plastic hardening and polynomial correction.
    
    sigma = E(T)*eps + (K1(T)*eps + K2(T)*eps^2) / (1 + H1(T)*eps) + C(T)*eps^3
    
    Params:
    p[0]: E0
    p[1]: c_E
    p[2]: K1_0
    p[3]: c_K1
    p[4]: K2_0
    p[5]: c_K2
    p[6]: h1_0 (log H1_0)
    p[7]: c_h1
    p[8]: C_0
    p[9]: c_C
    """
    eps = strain
    T = temp
    
    # Parameter unpacking
    E0 = params[0]
    c_E = params[1]
    K0 = params[2]
    c_K = params[3]
    alpha0 = params[4]
    c_alpha = params[5]
    C0 = params[6]
    c_C = params[7]
    D0 = params[8]
    c_D = params[9]
    
    # Temperature-dependent coefficients
    E_T = E0 * (1.0 + c_E * T)
    K_T = K0 * (1.0 + c_K * T)
    alpha_T = alpha0 * (1.0 + c_alpha * T)
    C_T = C0 * (1.0 + c_C * T)
    D_T = D0 * (1.0 + c_D * T)
    
    # Elastic term
    elastic = E_T * eps
    
    # Plastic term with exponential saturation
    # sigma_plastic = K_T * (1 - exp(-alpha_T * eps))
    # Clip the exponent argument to prevent overflow/underflow issues
    # alpha_T * eps can be large positive or negative.
    # If alpha_T * eps is large positive, exp(-large) -> 0.
    # If alpha_T * eps is large negative, exp(-large neg) = exp(large pos) -> inf.
    # So we clip the argument -alpha_T * eps.
    arg = -alpha_T * eps
    arg = np.clip(arg, -50, 50)
    plastic = K_T * (1.0 - np.exp(arg))
    
    # Polynomial corrections
    quadratic = C_T * (eps * eps)
    cubic = D_T * (eps * eps * eps)
    
    stress = elastic + plastic + quadratic + cubic
    
    return stress
