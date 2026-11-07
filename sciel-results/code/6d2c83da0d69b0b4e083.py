"""Stress-strain model for aluminium rod using tanh-based elastic-plastic transition."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a tanh-based smooth step to model elastic-plastic transition:
    - Linear elastic region at low strain
    - Smooth transition to plastic hardening
    - Saturation plateau at high strain
    - Temperature modulates yield stress and hardening
    
    Parameters:
        strain: array of strain values (input)
        temp: array of temperature values (input)
        params: array of 10 fitted constants
    
    Returns:
        array of predicted stress values
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent yield strain (where plastic flow begins)
    # T decreases yield strain (higher temp -> lower yield)
    eps_y = p0 + p1 * temp
    
    # Transition width for elastic-plastic transition
    # Positive width ensures smooth transition
    width = p2 + p3 * temp
    
    # Ensure width is positive to avoid numerical issues
    width = np.maximum(width, 0.01)
    
    # Elastic modulus (temperature-dependent)
    E = p4 + p5 * temp
    
    # Plastic hardening modulus
    H = p6 + p7 * temp
    
    # Saturation stress (plateau)
    sigma_sat = p8 + p9 * temp
    
    # Smooth step function: 0 for eps << eps_y, 1 for eps >> eps_y
    # Using tanh for smoothness
    x = (strain - eps_y) / width
    step = 0.5 * (1.0 + np.tanh(x))
    
    # Elastic stress component (active when step is small)
    sigma_elastic = E * strain
    
    # Plastic stress component: starts at yield, hardens, saturates
    # sigma_plastic = sigma_sat * (1 - exp(-H * max(strain - eps_y, 0)))
    # But we want smooth transition, so use the step function
    
    # Combined model:
    # For low strain: mostly elastic
    # For high strain: transitions to hardening behavior
    # sigma = (1-step) * sigma_elastic + step * sigma_plastic_hardening
    
    # Plastic hardening term: linear hardening that saturates
    eps_plastic = np.maximum(strain - eps_y, 0.0)
    sigma_hardening = sigma_sat * (1.0 - np.exp(-H * eps_plastic))
    
    # Ensure stress is continuous at yield point
    # At strain = eps_y: sigma_elastic = E * eps_y
    # sigma_hardening at eps_y = 0
    # We need to blend properly
    
    # Better approach: 
    # sigma = E * strain * (1 - step) + (E * eps_y + sigma_hardening) * step
    # This ensures continuity at eps_y
    
    sigma = E * strain * (1.0 - step) + (E * eps_y + sigma_hardening) * step
    
    return sigma
# EVOLVE-BLOCK-END
