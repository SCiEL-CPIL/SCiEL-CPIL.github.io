"""E. coli growth rate model with Monod, Gaussian temp/pH, and Haldane inhibition."""
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    
    Structure:
      mu = mu_max * (s / (Ks + s)) * exp(-0.5 * ((temp - T_opt)/sigma_T)^2) 
                 * exp(-0.5 * ((pH - pH_opt)/sigma_pH)^2) 
                 * b / (Kb + b + b^2/Ki)
    
    Parameters (10 total, centered for better optimization from ones):
      p0: log(mu_max)  -> mu_max = exp(p0), center ~log(1.55)≈0.44, so p0≈0.44
      p1: log(Ks)      -> Ks = exp(p1), center ~log(5)≈1.6, so p1≈1.6
      p2: T_opt - 20   -> T_opt = 20 + p2, range [1,40], center ~20
      p3: log(sigma_T) -> sigma_T = exp(p3), typical ~8, log(8)≈2.08
      p4: pH_opt - 7   -> pH_opt = 7 + p4, range [2,12], center ~7
      p5: log(sigma_pH)-> sigma_pH = exp(p5), typical ~2, log(2)≈0.69
      p6: log(Kb)      -> Kb = exp(p6), typical ~1, log(1)=0
      p7: log(Ki)      -> Ki = exp(p7), typical ~10, log(10)≈2.3
      p8: offset       -> additive constant
      p9: unused (reserved for future)
    
    All positive parameters are log-transformed to ensure positivity.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Positive parameters via exp
    mu_max = np.exp(p0)
    Ks = np.exp(p1)
    sigma_T = np.exp(p3)
    sigma_pH = np.exp(p5)
    Kb = np.exp(p6)
    Ki = np.exp(p7)
    
    # Optimal values (can be negative offsets)
    T_opt = 20.0 + p2
    pH_opt = 7.0 + p4
    
    # Monod term for substrate
    monod = s / (Ks + s)
    
    # Gaussian temperature response
    temp_resp = np.exp(-0.5 * ((temp - T_opt) / sigma_T) ** 2)
    
    # Gaussian pH response
    ph_resp = np.exp(-0.5 * ((pH - pH_opt) / sigma_pH) ** 2)
    
    # Haldane inhibition for population density
    # b / (Kb + b + b^2/Ki)
    haldane = b / (Kb + b + b**2 / Ki)
    
    # Combine
    mu = mu_max * monod * temp_resp * ph_resp * haldane + p8
    
    return mu
