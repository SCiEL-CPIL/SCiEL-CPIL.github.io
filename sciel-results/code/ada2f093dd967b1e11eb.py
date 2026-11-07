import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a physics-inspired model with elastic, yield, and hardening components.
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure positivity for physical parameters using exponential transformation
    # This helps BFGS stay in valid regions (modulus, stress, exponent > 0)
    E_base = np.exp(p0)
    sigma_y_base = np.exp(p3)
    H_base = np.exp(p5)
    n_exp = np.exp(p7)
    delta_smooth = np.exp(p9)
    
    # Temperature dependencies
    # Assume linear decrease in stiffness and strength with temp
    # Scaling factors to keep temp effects moderate
    temp_E_factor = 1.0 - 0.1 * p1 * temp
    temp_sigma_factor = 1.0 - 0.1 * p4 * temp
    temp_H_factor = 1.0 - 0.1 * p6 * temp
    
    # Effective properties
    E_eff = E_base * temp_E_factor
    sigma_y_eff = sigma_y_base * temp_sigma_factor
    H_eff = H_base * temp_H_factor
    
    # Plastic strain calculation using smooth hinge
    # epsilon_pl = max(0, strain - yield_strain) approximated smoothly
    diff = strain - p2
    epsilon_pl = 0.5 * (diff + np.sqrt(diff**2 + delta_smooth**2))
    
    # Yield transition term (smooth step)
    # This adds a stress jump/curve at the yield point
    yield_transition = sigma_y_eff * np.tanh(diff / delta_smooth)
    
    # Hardening term
    # Power law hardening
    # Handle potential issues if epsilon_pl is very small and n is non-integer
    # Since epsilon_pl >= 0 and n > 0, this is safe. 
    # If epsilon_pl is 0, 0^n is 0 (for n>0).
    hardening = H_eff * np.power(epsilon_pl, n_exp)
    
    # Elastic term
    elastic = E_eff * strain
    
    # Total stress
    stress = elastic + yield_transition + hardening + p8
    
    return stress
