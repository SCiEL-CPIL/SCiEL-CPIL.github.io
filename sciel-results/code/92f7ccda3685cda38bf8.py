import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model for E. coli growth rate with biologically motivated interactions.
    
    Features:
    1. Input normalization to ~[0, 1] for BFGS conditioning.
    2. Monod kinetics for substrate saturation.
    3. Quadratic responses for temperature, pH, and density.
    4. Substrate-temperature interaction (monod_s * T_norm).
    
    params (10):
      p[0]: bias
      p[1]: Monod scale
      p[2]: log-Ks (Ks = exp(p2 - 2))
      p[3]: T linear
      p[4]: T quadratic
      p[5]: pH linear
      p[6]: pH quadratic
      p[7]: b linear
      p[8]: b quadratic
      p[9]: substrate-temperature interaction coeff
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Normalize inputs to approximately [0, 1]
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    T_norm = (temp - 1.0) / 39.0
    pH_norm = (pH - 2.0) / 10.0
    
    # Monod kinetics for substrate
    Ks = np.exp(p2 - 2.0)
    monod_s = s_norm / (Ks + s_norm)
    
    # Construct prediction
    y = p0
    y += p1 * monod_s
    y += p3 * T_norm + p4 * (T_norm ** 2)
    y += p5 * pH_norm + p6 * (pH_norm ** 2)
    y += p7 * b_norm + p8 * (b_norm ** 2)
    y += p9 * (monod_s * T_norm)
    
    return y
