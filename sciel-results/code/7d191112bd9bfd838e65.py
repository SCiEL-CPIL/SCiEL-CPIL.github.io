"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: sigma = (p0 + p1*strain) * (p2 + p3*temp) * (1 + p4*strain^p5)
                    / (1 + p6*strain^p7 + p8*temp^p9) + p10
    
    This multiplicative interaction model decouples:
    - Linear elastic scaling: (p0 + p1*strain)
    - Temperature-dependent stiffness: (p2 + p3*temp)
    - Non-linear strain saturation: (1 + p4*strain^p5) / (1 + p6*strain^p7 + p8*temp^p9)
    - Offset: p10
    """
    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = params
    
    # Compute strain powers safely
    # Use abs for fractional powers to avoid NaN with negative strain
    strain_abs = np.abs(strain)
    
    # Linear elastic term
    elastic = p0 + p1 * strain
    
    # Temperature-dependent stiffness
    temp_factor = p2 + p3 * temp
    
    # Non-linear strain saturation numerator
    strain_sat_num = 1.0 + p4 * np.power(strain_abs, p5)
    
    # Non-linear strain saturation denominator
    strain_sat_den = 1.0 + p6 * np.power(strain_abs, p7) + p8 * np.power(np.abs(temp), p9)
    
    # Avoid division by zero
    strain_sat_den = np.where(np.abs(strain_sat_den) < 1e-12, 1e-12, strain_sat_den)
    
    # Combine all factors
    stress = elastic * temp_factor * strain_sat_num / strain_sat_den + p10
    
    return stress
# EVOLVE-BLOCK-END
