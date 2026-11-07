"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a rational function for the elastic-plastic transition combined
    with polynomial terms for strain hardening and temperature effects.
    
    Model:
    E(T) = p0 + p1 * T
    stress = (E(T) * strain) / (1 + p2 * strain) + p3 * strain^2 + p4 * strain^3 
             + p5 * T * strain + p6 * T^2 * strain + p7 * T + p8 * T^2 + p9
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are float64 for precision
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Calculate Temperature-dependent Elastic Modulus
    # E(T) = p0 + p1 * T
    E_T = p0 + p1 * temp
    
    # Rational term for elastic-plastic transition
    # This term starts linear (slope E_T) and saturates as strain increases
    # Adding a small epsilon to denominator to prevent division by zero if p2 becomes large negative
    denom = 1.0 + p2 * strain + 1e-8
    rational_term = (E_T * strain) / denom
    
    # Polynomial terms for strain hardening
    strain_sq = strain * strain
    strain_cu = strain_sq * strain
    hardening = p3 * strain_sq + p4 * strain_cu
    
    # Temperature interaction terms
    temp_sq = temp * temp
    temp_interact = p5 * temp * strain + p6 * temp_sq * strain
    
    # Temperature offset terms
    temp_offset = p7 * temp + p8 * temp_sq
    
    # Constant offset
    offset = p9
    
    # Total stress
    stress = rational_term + hardening + temp_interact + temp_offset + offset
    
    return stress
# EVOLVE-BLOCK-END
