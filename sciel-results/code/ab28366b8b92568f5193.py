"""E. coli growth rate model: Monod substrate x Gaussian temp/pH with flexibility."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model with biologically-motivated multiplicative structure.
    
    mu = p0 * (s / (p1 + s)) * exp(-0.5 * ((temp - p2) / p3)^2) * exp(-0.5 * ((pH - p4) / p5)^2) * (1 + p6 * b_norm + p7 * b_norm^2)
    
    Parameters (10):
    p0: Maximum growth rate (scale)
    p1: Monod half-saturation constant Ks
    p2: Optimal temperature
    p3: Temperature width (sigma)
    p4: Optimal pH
    p5: pH width (sigma)
    p6: Linear coefficient for b (normalized)
    p7: Quadratic coefficient for b (normalized)
    p8: Additional temp-pH interaction coupling
    p9: Residual linear temp term (for asymmetry)
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Safe parameter handling to avoid division by zero or negative widths
    Ks = np.abs(p1) + 1e-6
    sigma_t = np.abs(p3) + 1e-6
    sigma_p = np.abs(p5) + 1e-6
    
    # Monod term for substrate
    monod = s / (Ks + s)
    
    # Gaussian response for temperature
    temp_z = (temp - p2) / sigma_t
    temp_gauss = np.exp(-0.5 * temp_z ** 2)
    
    # Gaussian response for pH
    ph_z = (pH - p4) / sigma_p
    ph_gauss = np.exp(-0.5 * ph_z ** 2)
    
    # Normalized population density (b in [1, 5] -> [0, 1])
    b_norm = (b - 1.0) / 4.0
    
    # Density-dependent modification (linear + quadratic)
    density_factor = 1.0 + p6 * b_norm + p7 * b_norm ** 2
    
    # Interaction and asymmetry terms
    # Coupling between temp and pH deviations
    interaction = p8 * temp_z * ph_z
    # Linear temp asymmetry term
    asym = p9 * temp_z
    
    # Combine: main multiplicative + correction terms
    pred = p0 * monod * temp_gauss * ph_gauss * density_factor * np.exp(interaction + asym)
    
    # Ensure finite output
    return np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
