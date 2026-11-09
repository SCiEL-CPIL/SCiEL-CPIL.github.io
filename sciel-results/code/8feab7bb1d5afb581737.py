import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Mechanistic-inspired model for E. coli growth rate.
    
    Structure:
    - Monod kinetics for substrate: s / (Ks + s)
    - Temperature dependence: Gaussian around optimal T
    - pH dependence: Gaussian around optimal pH
    - Density dependence: logistic inhibition
    - Multiplicative interaction of factors
    
    params layout (10 constants):
      p[0]: max growth rate (mu_max)
      p[1]: Monod half-saturation constant (Ks)
      p[2]: optimal temperature (T_opt)
      p[3]: temperature width (sigma_T)
      p[4]: optimal pH (pH_opt)
      p[5]: pH width (sigma_pH)
      p[6]: density carrying capacity (K)
      p[7]: density inhibition coefficient (alpha)
      p[8]: bias term
      p[9]: scaling factor
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
    
    # Monod kinetics for substrate
    monod = s / (p1 + s)
    
    # Temperature dependence (Gaussian)
    temp_factor = np.exp(-0.5 * ((temp - p2) / p3) ** 2)
    
    # pH dependence (Gaussian)
    ph_factor = np.exp(-0.5 * ((pH - p4) / p5) ** 2)
    
    # Density dependence (logistic inhibition)
    density_factor = 1.0 / (1.0 + p7 * (b / p6))
    
    # Combine all factors multiplicatively
    y = p0 * monod * temp_factor * ph_factor * density_factor
    
    # Add bias and scaling
    y = p9 * y + p8
    
    return y
