"""Symbolic regression for E. coli growth rate using a mechanistic Monod model."""
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model based on Monod kinetics with temperature and pH optima.
    
    Parameters:
    b: population density
    s: substrate concentration
    temp: temperature
    pH: pH level
    params: array of 10 fitted constants
    
    Returns:
    Predicted growth rate
    """
    # Extract parameters with safe defaults to avoid numerical issues
    p0 = params[0]  # max growth rate
    p1 = params[1]  # half-saturation constant (Ks)
    p2 = params[2]  # temperature optimum
    p3 = params[3]  # temperature width
    p4 = params[4]  # pH optimum
    p5 = params[5]  # pH width
    p6 = params[6]  # carrying capacity / density limit
    p7 = params[7]  # density scaling
    p8 = params[8]  # additional scaling
    p9 = params[9]  # offset
    
    # Ensure positive parameters for biological quantities
    Ks = np.abs(p1) + 1e-6
    T_width = np.abs(p3) + 1e-6
    pH_width = np.abs(p5) + 1e-6
    carrying = np.abs(p6) + 1e-6
    
    # Monod term for substrate
    monod = s / (Ks + s)
    
    # Temperature response: Gaussian around optimum
    T_term = np.exp(-((temp - p2) ** 2) / (2 * T_width ** 2))
    
    # pH response: Gaussian around optimum
    pH_term = np.exp(-((pH - p4) ** 2) / (2 * pH_width ** 2))
    
    # Density limitation: logistic-like term
    # At low density, growth is maximal; at high density, it decreases
    density_term = 1.0 / (1.0 + np.exp(p7 * (b - carrying)))
    
    # Combine all terms
    growth = p0 * monod * T_term * pH_term * density_term * p8 + p9
    
    # Ensure finite output
    growth = np.where(np.isfinite(growth), growth, 0.0)
    
    return growth
