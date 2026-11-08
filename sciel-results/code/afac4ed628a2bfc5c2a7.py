import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    
    Inputs:
    b: population density
    s: substrate concentration
    temp: temperature
    pH: pH level
    params: array of 10 parameters
    
    Model:
    y = mu_max * Monod(s) * Exp_T(T) * Exp_pH(pH) * Inhibition(b) + offset
    
    Params:
    p0: K_s (Monod half-saturation, squared for positivity)
    p1, p2, p3: Log-Quadratic coefficients for Temperature (c0, c1, c2)
    p4, p5, p6: Log-Quadratic coefficients for pH (c0, c1, c2)
    p7: K_b (Density inhibition constant, squared for positivity)
    p8: mu_max (Maximum growth rate)
    p9: offset (Baseline)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure positivity for scale parameters
    K_s = p0**2 + 1e-6
    K_b = p7**2 + 1e-6
    
    # Monod substrate kinetics
    # s / (K_s + s)
    monod_s = s / (K_s + s)
    
    # Temperature effect: exp(p1 + p2*T + p3*T^2)
    # This allows a peak if p3 < 0
    exp_T = np.exp(p1 + p2 * temp + p3 * temp**2)
    
    # pH effect: exp(p4 + p5*pH + p6*pH^2)
    exp_pH = np.exp(p4 + p5 * pH + p6 * pH**2)
    
    # Density inhibition: 1 / (1 + b / K_b)
    inhib_b = 1.0 / (1.0 + b / K_b)
    
    # Combine
    # mu_max * monod_s * exp_T * exp_pH * inhib_b + offset
    y = p8 * monod_s * exp_T * exp_pH * inhib_b + p9
    
    return y
