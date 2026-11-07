import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using Monod kinetics for substrate,
    Gaussian responses for temperature and pH, and a smooth inhibition term for density.
    
    Parameters:
    p0: max growth rate multiplier
    p1: substrate half-saturation constant (Ks)
    p2: temperature optimum offset from 37
    p3: temperature width (sigma)
    p4: pH optimum offset from 7
    p5: pH width (sigma)
    p6: density inhibition constant (Kb)
    p7: density inhibition strength
    p8: temperature-pH interaction correction (linear in centered vars)
    p9: bias
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure positive constants for stability
    Ks = np.abs(p1) + 0.1
    sigma_T = np.abs(p3) + 0.5
    sigma_pH = np.abs(p5) + 0.5
    Kb = np.abs(p6) + 0.1
    
    # Centered variables for better conditioning
    T_centered = temp - (37.0 + p2)
    pH_centered = pH - (7.0 + p4)
    
    # Monod substrate term: S / (Ks + S)
    # Clip S to avoid overflow if S is huge, though range is [1, 40]
    monod_s = s / (Ks + s)
    
    # Gaussian environmental terms
    # Using exp(-x^2 / (2*sigma^2))
    gauss_T = np.exp(-(T_centered ** 2) / (2.0 * sigma_T ** 2))
    gauss_pH = np.exp(-(pH_centered ** 2) / (2.0 * sigma_pH ** 2))
    
    # Density inhibition: 1 - (b / (Kb + b)) * strength
    # This goes from 1 (low density) to 1-strength (high density)
    # If strength is 1, it goes to 0.
    # Let's use a smooth logistic-like decay or simple rational.
    # Inhibition factor: 1 / (1 + (b/Kb)^2) is also smooth.
    # Let's use: 1 - p7 * (b / (Kb + b))
    # This is bounded between 1-p7 and 1.
    inhibition = 1.0 - p7 * (b / (Kb + b))
    
    # Interaction term: small correction based on T and pH deviations
    interaction = 1.0 + p8 * (T_centered * pH_centered) / (sigma_T * sigma_pH)
    
    # Combine
    growth = p0 * monod_s * gauss_T * gauss_pH * inhibition * interaction + p9
    
    return growth
