import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model combining Monod substrate kinetics,
    Gaussian temperature and pH response, and exponential density inhibition.
    
    Parameters:
        p0: max growth rate scale
        p1: Monod half-saturation constant (Ks)
        p2: optimal temperature
        p3: temperature sensitivity (width)
        p4: optimal pH
        p5: pH sensitivity (width)
        p6: density inhibition strength
        p7: temperature offset correction
        p8: pH offset correction
        p9: baseline offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure positive parameters for physical quantities
    Ks = np.abs(p1) + 1e-6
    T_opt = p2 + p7
    T_width = np.abs(p3) + 1e-6
    pH_opt = p4 + p8
    pH_width = np.abs(p5) + 1e-6
    inhibition = np.abs(p6)
    
    # Monod kinetics for substrate
    monod = s / (Ks + s)
    
    # Gaussian response for temperature
    temp_resp = np.exp(-((temp - T_opt) ** 2) / (2.0 * T_width ** 2))
    
    # Gaussian response for pH
    ph_resp = np.exp(-((pH - pH_opt) ** 2) / (2.0 * pH_width ** 2))
    
    # Exponential inhibition for population density
    density_inhibition = np.exp(-inhibition * b)
    
    # Combine all factors
    growth = p0 * monod * temp_resp * ph_resp * density_inhibition + p9
    
    return growth
