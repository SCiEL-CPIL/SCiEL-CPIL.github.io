import numpy as np

def equation(b, s, temp, pH, params):
    """
    Biologically motivated multiplicative model for E. coli growth rate.
    
    The model assumes the growth rate is the product of:
    1. A saturation term for substrate (Monod-like): s / (Ks + s)
    2. A temperature response function (Gaussian bell): exp(-((temp - Topt)^2) / (2 * sigma_t^2))
    3. A pH response function (Quadratic): 1 - ((pH - pHopt)^2 / scale_pH)
    4. A density-dependent inhibition term (Logistic-like): 1 / (1 + b / K_b)
    
    This structure is physically plausible for microbial growth and allows the 
    optimizer to find meaningful physiological parameters rather than arbitrary 
    polynomial coefficients.
    
    Inputs:
    b: population density
    s: substrate concentration
    temp: temperature
    pH: pH level
    
    Params (10 constants):
    p0: max growth rate (mu_max)
    p1: substrate half-saturation constant (Ks)
    p2: optimal temperature (Topt)
    p3: temperature width (sigma_t)
    p4: optimal pH (pHopt)
    p5: pH width (scale_pH)
    p6: carrying capacity (K_b)
    p7: substrate affinity modifier (optional, to adjust Monod shape)
    p8: temperature sharpness modifier (optional)
    p9: global scaling/bias
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure positive values for denominators and widths to avoid division by zero or invalid exp
    # Using softplus-like or simple clipping for stability
    Ks = np.maximum(p1, 1e-6)
    sigma_t = np.maximum(p3, 1e-6)
    scale_pH = np.maximum(p5, 1e-6)
    K_b = np.maximum(p6, 1e-6)
    
    # 1. Substrate effect: Monod equation
    # s / (Ks + s)
    # p7 can modulate the saturation level or shape
    monod_term = s / (Ks + s)
    # Optionally allow for a slight deviation from standard Monod if p7 != 1
    # For now, keep it standard but scaled by p7 if needed? 
    # Let's stick to standard Monod for robustness, p7 can be used for global scaling if needed.
    # Actually, let's use p7 to adjust the half-saturation constant dynamically? No, p1 is Ks.
    # Let's use p7 as a multiplier for the substrate term to allow for non-standard saturation.
    substrate_factor = monod_term * (1.0 + 0.1 * p7) # Small modulation
    
    # 2. Temperature effect: Gaussian
    # exp(-((temp - Topt)^2) / (2 * sigma_t^2))
    # p8 can adjust the sharpness
    temp_diff = temp - p2
    temp_factor = np.exp(-(temp_diff ** 2) / (2.0 * sigma_t ** 2 * (1.0 + 0.1 * p8)))
    
    # 3. pH effect: Quadratic decline from optimum
    # 1 - ((pH - pHopt)^2 / scale_pH)
    # Clip to 0 to ensure non-negative growth rate contribution
    pH_diff = pH - p4
    pH_factor = np.maximum(1.0 - (pH_diff ** 2) / scale_pH, 0.0)
    
    # 4. Density effect: Logistic inhibition
    # 1 / (1 + b / K_b)
    density_factor = 1.0 / (1.0 + b / K_b)
    
    # Combine
    # y = p0 * substrate_factor * temp_factor * pH_factor * density_factor
    # p9 is a global scale factor
    
    y = p0 * substrate_factor * temp_factor * pH_factor * density_factor * p9
    
    return y
