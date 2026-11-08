import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: Linear Elastic base with Tanh Hardening and Polynomial Corrections.
    sigma = E(T) * strain * (1 + A(T)*tanh(B(T)*strain) + C(T)*strain^2 + D(T)*strain^3)
    
    Params:
    p[0]: E0 (Young's modulus scale)
    p[1]: c_E (Temp coeff for E)
    p[2]: A0 (Tanh amplitude base)
    p[3]: A1 (Tanh amplitude temp)
    p[4]: B0 (Tanh steepness base)
    p[5]: B1 (Tanh steepness temp)
    p[6]: C0 (Poly quad base)
    p[7]: C1 (Poly quad temp)
    p[8]: D0 (Poly cubic base)
    p[9]: D1 (Poly cubic temp)
    """
    # Extract params
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
    
    # Temperature dependent coefficients
    T = temp
    E_T = E0 * (1.0 + c_E * T)
    
    A_T = A0 + A1 * T
    B_T = B0 + B1 * T
    C_T = C0 + C1 * T
    D_T = D0 + D1 * T
    
    eps = strain
    
    # Tanh hardening term
    # Stable for all real values
    hardening_tanh = A_T * np.tanh(B_T * eps)
    
    # Polynomial corrections
    # Note: These are multiplied by eps in the final stress calculation,
    # so they contribute eps^2 and eps^3 terms to the stress.
    poly_corr = C_T * eps + D_T * (eps * eps)
    
    # Total hardening factor
    factor = 1.0 + hardening_tanh + poly_corr
    
    # Stress
    stress = E_T * eps * factor
    
    return stress
