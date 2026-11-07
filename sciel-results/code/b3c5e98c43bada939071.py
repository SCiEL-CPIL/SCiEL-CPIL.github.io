"""E. coli growth rate symbolic regression with structural model."""
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using Monod kinetics, temperature/pH response curves,
    and biomass inhibition.
    
    Parameters (10 total):
    p[0]: Max growth rate amplitude
    p[1]: Monod half-saturation constant Ks
    p[2]: Temperature optimal value
    p[3]: Temperature width parameter
    p[4]: pH optimal value
    p[5]: pH width parameter
    p[6]: Biomass inhibition coefficient
    p[7]: Substrate inhibition coefficient
    p[8]: Temperature skew coefficient
    p[9]: pH skew coefficient
    """
    # Clip parameters to reasonable ranges to avoid numerical issues
    p0 = params[0]
    p1 = np.maximum(params[1], 1e-6)  # Ks must be positive
    p2 = params[2]
    p3 = np.maximum(params[3], 0.1)   # Temperature width
    p4 = params[4]
    p5 = np.maximum(params[5], 0.1)   # pH width
    p6 = params[6]
    p7 = np.maximum(params[7], 0.0)   # Substrate inhibition
    p8 = params[8]
    p9 = params[9]
    
    # Monod kinetics for substrate
    monod = s / (p1 + s)
    
    # Substrate inhibition (Haldane)
    inhibition = 1.0 / (1.0 + p7 * s)
    
    # Temperature response: asymmetric Gaussian
    temp_diff = temp - p2
    temp_response = np.exp(-0.5 * (temp_diff / p3) ** 2) * (1.0 + p8 * temp_diff)
    # Ensure positive
    temp_response = np.maximum(temp_response, 1e-10)
    
    # pH response: asymmetric Gaussian
    ph_diff = pH - p4
    ph_response = np.exp(-0.5 * (ph_diff / p5) ** 2) * (1.0 + p9 * ph_diff)
    # Ensure positive
    ph_response = np.maximum(ph_response, 1e-10)
    
    # Biomass inhibition (exponential decay)
    biomass_inhibition = np.exp(-p6 * b)
    
    # Combine all factors
    growth_rate = p0 * monod * inhibition * temp_response * ph_response * biomass_inhibition
    
    return growth_rate
