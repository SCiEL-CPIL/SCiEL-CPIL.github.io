# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod.
    
    Uses a smooth yield transition with plastic hardening and temperature effects.
    The model captures:
    - Elastic linear region with temperature-dependent modulus
    - Smooth yield transition (tanh-based)
    - Plastic hardening (quadratic in plastic strain)
    - Temperature effects on yield strength and hardening
    
    params (10 values):
      p[0]: elastic modulus scale
      p[1]: yield strain threshold
      p[2]: yield stress scale
      p[3]: hardening coefficient
      p[4]: temperature effect on yield
      p[5]: temperature effect on modulus
      p[6]: cubic hardening
      p[7]: interaction term
      p[8]: offset
      p[9]: smoothness parameter for yield transition
    """
    s = strain
    t = temp
    p = params
    
    # Normalize inputs to [0, 1] range for better conditioning
    s_min, s_max = -0.001911, 0.9636
    t_min, t_max = 0.06667, 1.0
    
    s_norm = (s - s_min) / (s_max - s_min)  # [0, 1]
    t_norm = (t - t_min) / (t_max - t_min)  # [0, 1]
    
    # Elastic modulus (temperature dependent)
    E = p[0] * (1.0 + p[5] * t_norm)
    
    # Yield strain (temperature dependent)
    s_yield = p[1] * (1.0 + p[4] * t_norm)
    
    # Smooth yield transition using tanh
    # This creates a smooth transition from elastic to plastic
    yield_arg = (s - s_yield) / (0.01 + abs(p[9]))
    yield_factor = 0.5 * (1.0 + np.tanh(yield_arg))
    
    # Elastic stress
    stress_elastic = E * s
    
    # Plastic strain (positive part, smoothed)
    s_plastic = np.maximum(s - s_yield, 0.0)
    
    # Yield stress (temperature dependent)
    sigma_y = p[2] * (1.0 - 0.3 * t_norm)
    
    # Plastic hardening
    hardening = p[3] * s_plastic + p[6] * s_plastic**2
    
    # Total plastic stress contribution
    stress_plastic = sigma_y + hardening
    
    # Combine elastic and plastic with smooth transition
    stress = stress_elastic * (1.0 - yield_factor) + stress_plastic * yield_factor
    
    # Temperature interaction effects
    stress = stress + p[7] * t_norm * s * (1.0 - yield_factor)
    
    # Offset
    stress = stress + p[8]
    
    return stress
# EVOLVE-BLOCK-END
