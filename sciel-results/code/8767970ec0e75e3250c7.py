import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod, linear in parameters.
    sigma = sum_i p_i * basis_i(strain, temp)
    
    Basis functions chosen to capture:
    - Elastic linear region
    - Yield transition (smooth)
    - Plastic hardening
    - Temperature softening effects
    """
    eps = strain
    T = temp
    p = params
    
    # Temperature factor: normalizes temp to [0,1] range roughly
    Tn = (T - 0.06667) / 0.93333
    
    # Basis functions (linear in params, nonlinear in inputs)
    b0 = np.ones_like(eps)
    b1 = eps
    b2 = eps**2
    b3 = eps**3
    b4 = np.tanh(eps / 0.01)
    b5 = np.tanh(eps / 0.05)
    b6 = eps * Tn
    b7 = eps**2 * Tn
    b8 = np.tanh(eps / 0.01) * Tn
    b9 = eps * np.exp(-eps / 0.5)
    
    sigma = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
             p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
             p[8] * b8 + p[9] * b9)
    
    return sigma
