import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-inspired stress-strain model for an aluminium rod with temperature dependence.
    
    Uses a smooth elastic-plastic model with:
    - Temperature-dependent Young's modulus
    - Temperature-dependent yield strain
    - Smooth transition via tanh (differentiable for BFGS)
    - Linear + quadratic plastic hardening
    - Temperature effects on yield stress and hardening
    
    10 parameters:
    p[0]: base Young's modulus (E0)
    p[1]: temp coefficient for E
    p[2]: base yield strain (eps_y0)
    p[3]: temp coefficient for yield strain
    p[4]: transition sharpness
    p[5]: linear hardening modulus
    p[6]: quadratic hardening coefficient
    p[7]: temp coefficient for yield stress
    p[8]: temp coefficient for hardening
    p[9]: offset/intercept
    """
    s = strain
    t = temp
    p = params
    
    # Normalized temperature: 0 at t_min, 1 at t_max
    t_norm = (t - 0.06667) / (1.0 - 0.06667)
    t_norm = np.clip(t_norm, 0.0, 1.0)
    
    # Temperature-dependent Young's modulus
    E = p[0] * (1.0 + p[1] * t_norm)
    E = np.maximum(E, 1e-6)  # Keep positive
    
    # Temperature-dependent yield strain
    eps_y = p[2] * (1.0 + p[3] * t_norm)
    eps_y = np.maximum(eps_y, 1e-8)  # Keep positive
    
    # Elastic stress
    sigma_elastic = E * s
    
    # Plastic strain (positive when s > eps_y)
    eps_plastic = np.maximum(s - eps_y, 0.0)
    
    # Smooth transition: use tanh to smoothly switch from elastic to plastic
    # At s < eps_y: transition ~ 0, at s > eps_y: transition ~ 1
    delta = s - eps_y
    sharpness = np.abs(p[4]) + 1.0
    transition = 0.5 * (1.0 + np.tanh(sharpness * delta))
    
    # Plastic stress: linear + quadratic hardening
    hardening_linear = p[5] * eps_plastic
    hardening_quadratic = p[6] * eps_plastic * eps_plastic
    
    # Temperature effects on plastic behavior
    yield_stress = E * eps_y  # stress at yield
    hardening_temp_mod = 1.0 + p[8] * t_norm
    plastic_stress = yield_stress + (hardening_linear + hardening_quadratic) * hardening_temp_mod
    
    # Combine elastic and plastic with smooth transition
    # When transition=0: pure elastic, when transition=1: elastic + plastic
    # Actually, let's think about this differently.
    # In elastic region: stress = E * s
    # In plastic region: stress = E * eps_y + hardening(eps - eps_y)
    # Use smooth interpolation
    
    sigma = transition * plastic_stress + (1.0 - transition) * sigma_elastic
    
    # Temperature effect on overall stress level
    sigma = sigma * (1.0 + p[7] * t_norm * 0.1)  # small temp effect
    
    # Add offset
    sigma = sigma + p[9]
    
    # Ensure finite output
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return sigma
