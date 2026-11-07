"""E. coli growth rate model with Monod kinetics, Gaussian temp/pH responses, and density inhibition."""
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate from population density, substrate, temperature and pH.
    
    Model structure:
    - Substrate: Monod kinetics S/(Ks + S)
    - Temperature: Gaussian response around optimal temp
    - pH: Gaussian response around optimal pH
    - Population density: exponential inhibition (crowding)
    - Plus linear interaction terms for non-separable effects
    
    10 parameters:
    p0: max growth rate multiplier
    p1: Monod half-saturation constant (Ks)
    p2: temperature optimum offset (from 20)
    p3: temperature width
    p4: pH optimum offset (from 7)
    p5: pH width
    p6: density inhibition constant
    p7: linear temp coefficient (interaction)
    p8: linear pH coefficient (interaction)
    p9: intercept
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Safe parameter transformations to keep values in reasonable ranges
    # Ks should be positive
    Ks = np.abs(p1) + 0.01
    
    # Temperature optimum: center around 20, allow shift
    T_opt = 20.0 + p2
    # Temperature width: keep positive, reasonable range
    T_width = np.abs(p3) + 0.5
    
    # pH optimum: center around 7, allow shift
    pH_opt = 7.0 + p4
    # pH width: keep positive
    pH_width = np.abs(p5) + 0.5
    
    # Density inhibition constant: keep positive
    K_inh = np.abs(p6) + 0.01
    
    # Monod kinetics for substrate
    monod = s / (Ks + s)
    
    # Gaussian response for temperature
    temp_resp = np.exp(-((temp - T_opt) ** 2) / (2.0 * T_width ** 2))
    
    # Gaussian response for pH
    pH_resp = np.exp(-((pH - pH_opt) ** 2) / (2.0 * pH_width ** 2))
    
    # Exponential inhibition for population density (crowding)
    density_inh = np.exp(-b / K_inh)
    
    # Core multiplicative model
    core = p0 * monod * temp_resp * pH_resp * density_inh
    
    # Additive interaction terms to capture non-separable effects
    # These help BFGS converge better by providing gradient paths
    interaction = p7 * temp * monod + p8 * pH * monod + p9
    
    return core + interaction
