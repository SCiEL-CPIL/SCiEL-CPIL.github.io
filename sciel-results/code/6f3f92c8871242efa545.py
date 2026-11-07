"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using Monod kinetics with temperature and pH optima.
    
    The model captures:
    - Substrate limitation via Monod term: s / (Ks + s)
    - Temperature effect via Gaussian: exp(-((T - T_opt)^2) / (2 * sigma_T^2))
    - pH effect via Gaussian: exp(-((pH - pH_opt)^2) / (2 * sigma_pH^2))
    - Density inhibition: 1 / (1 + (b / Kb)^n)
    
    Parameters:
    b: Population density
    s: Substrate concentration
    temp: Temperature
    pH: pH level
    params: Array of 10 fitted constants.
        p[0]: Max growth rate scale (mu_max)
        p[1]: Log of Monod half-saturation constant (Ks)
        p[2]: Optimal temperature
        p[3]: Log of temperature width (sigma_T)
        p[4]: Optimal pH
        p[5]: Log of pH width (sigma_pH)
        p[6]: Log of density inhibition constant (Kb)
        p[7]: Density inhibition exponent
        p[8]: Offset/bias
        p[9]: Additional scaling factor
    """
    # Extract parameters
    mu_max = params[0]
    Ks = np.exp(params[1])  # Half-saturation constant for substrate
    T_opt = params[2]       # Optimal temperature
    sigma_T = np.exp(params[3])  # Temperature width
    pH_opt = params[4]      # Optimal pH
    sigma_pH = np.exp(params[5])  # pH width
    Kb = np.exp(params[6])  # Density inhibition constant
    n = params[7]           # Density inhibition exponent
    offset = params[8]      # Bias term
    scale = params[9]       # Additional scaling
    
    # Ensure positive values for widths and constants
    sigma_T = np.maximum(sigma_T, 1e-6)
    sigma_pH = np.maximum(sigma_pH, 1e-6)
    Kb = np.maximum(Kb, 1e-6)
    Ks = np.maximum(Ks, 1e-6)
    
    # Monod term for substrate
    monod_s = s / (Ks + s)
    
    # Temperature effect (Gaussian)
    temp_effect = np.exp(-((temp - T_opt) ** 2) / (2 * sigma_T ** 2))
    
    # pH effect (Gaussian)
    pH_effect = np.exp(-((pH - pH_opt) ** 2) / (2 * sigma_pH ** 2))
    
    # Density inhibition
    density_inhibition = 1.0 / (1.0 + (b / Kb) ** n)
    
    # Combine all terms
    growth = mu_max * monod_s * temp_effect * pH_effect * density_inhibition * scale + offset
    
    return growth
# EVOLVE-BLOCK-END
