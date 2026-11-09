"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Combines a tanh-based elastic-plastic model with temperature-dependent
    coefficients and polynomial corrections.
    
    Model:
      sigma = E(T)*eps + S(T)*tanh(k*(eps - eps_y(T))) + H(T)*eps*tanh(k*(eps - eps_y(T))) + Q*eps^2 + offsets
    
    Where:
      E(T) = p0 + p1*T
      S(T) = p2 + p3*T
      H(T) = p7 + p8*T
      k    = p4
      eps_y(T) = p5*T + p6
      Q    = p9
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent coefficients
    elastic_modulus = p0 + p1 * temp
    yield_amplitude = p2 + p3 * temp
    hardening_slope = p7 + p8 * temp
    
    # Yield strain (temperature-dependent)
    yield_strain = p5 * temp + p6
    
    # Transition argument for tanh
    # Shifted by yield strain so the transition occurs at the yield point
    transition_arg = p4 * (strain - yield_strain)
    tanh_term = np.tanh(transition_arg)
    
    # Assemble stress
    # Elastic linear term
    elastic = elastic_modulus * strain
    
    # Yield plateau contribution (saturates)
    yield_contrib = yield_amplitude * tanh_term
    
    # Plastic hardening contribution (linear in strain, gated by tanh)
    hardening_contrib = hardening_slope * strain * tanh_term
    
    # Quadratic correction for nonlinearity
    quad = p9 * strain * strain
    
    result = elastic + yield_contrib + hardening_contrib + quad
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
# EVOLVE-BLOCK-END
