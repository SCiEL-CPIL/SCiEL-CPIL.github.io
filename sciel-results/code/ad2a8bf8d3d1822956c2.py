"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate using Monod kinetics with temperature and pH modulation."""
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Monod-like substrate saturation: s / (Ks + s)
    Ks = np.abs(p0) + 1e-6
    monod = s / (Ks + s)
    
    # Temperature response: Gaussian-like around optimum
    # temp is in [1, 12], likely normalized. Optimum around 37C -> ~0.7 of range?
    # Use a smooth peak: exp(-((temp - T_opt)^2) / (2 * sigma_T^2))
    T_opt = p1  # fitted optimum temperature (in input units)
    sigma_T = np.abs(p2) + 1e-6
    temp_factor = np.exp(-((temp - T_opt) ** 2) / (2 * sigma_T ** 2))
    
    # pH response: Gaussian-like around optimum pH
    pH_opt = p3  # fitted optimum pH
    sigma_pH = np.abs(p4) + 1e-6
    pH_factor = np.exp(-((pH - pH_opt) ** 2) / (2 * sigma_pH ** 2))
    
    # Density effect: possibly inhibitory at high density
    # Use a simple form: 1 - (b / b_max)^n, or just linear inhibition
    # b is in [1, 5]. Let's use: 1 / (1 + (b / b_half)^n)
    b_half = np.abs(p5) + 1e-6
    n_exp = np.abs(p6) + 1e-6
    density_factor = 1.0 / (1.0 + (b / b_half) ** n_exp)
    
    # Combine: max_rate * monod * temp_factor * pH_factor * density_factor
    max_rate = p7
    
    # Add a small baseline to avoid exact zeros
    baseline = p8
    
    # Optional: add a small linear correction term for flexibility
    correction = p9 * (b / 5.0)  # normalized small correction
    
    result = max_rate * monod * temp_factor * pH_factor * density_factor + baseline + correction
    
    return result
# EVOLVE-BLOCK-END
