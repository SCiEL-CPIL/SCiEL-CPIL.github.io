"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a Ramberg-Osgood type model with temperature-dependent coefficients:
    stress = E(T) * strain + A(T) * (|strain| + eps0)**n(T) * sign(strain)
    where E(T), A(T), n(T) are linear in temperature.
    
    Parameters (10 total):
    p0, p1: E(T) = p0 + p1 * T
    p2, p3: A(T) = p2 + p3 * T
    p4, p5: n(T) = p4 + p5 * T
    p6: offset strain eps0 (small positive to avoid singularity)
    p7: additional linear temp term on stress
    p8: quadratic temp term on stress
    p9: constant offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * temp
    
    # Temperature-dependent amplitude
    A = p2 + p3 * temp
    
    # Temperature-dependent exponent
    n = p4 + p5 * temp
    
    # Safe absolute strain with small offset to avoid singularity at zero
    abs_strain = np.abs(strain)
    safe_abs_strain = abs_strain + p6
    
    # Power term - clip exponent to avoid extreme values
    n_clipped = np.clip(n, 0.1, 10.0)
    power_term = np.power(safe_abs_strain, n_clipped)
    
    # Sign function
    sign_strain = np.sign(strain)
    
    # Main stress components
    elastic = E * strain
    plastic = A * power_term * sign_strain
    
    # Additional temperature-dependent terms
    temp_linear = p7 * temp
    temp_quadratic = p8 * temp * temp
    
    # Total stress
    stress = elastic + plastic + temp_linear + temp_quadratic + p9
    
    return stress
# EVOLVE-BLOCK-END
