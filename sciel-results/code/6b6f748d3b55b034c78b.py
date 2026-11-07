import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod using a smooth bilinear elastic-plastic
    model with temperature dependence and higher-order corrections.
    
    Model:
      - Elastic region: stress = E(T) * strain
      - Smooth transition to plastic region using a smoothstep-like function
      - Plastic region: stress = Y(T) + H(T) * (strain - epsilon_y(T))
      - Temperature dependence on all key parameters
      - Additional polynomial terms for fine-tuning
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent parameters
    # Elastic modulus: E(T) = p0 * (1 + p3 * temp)
    E = p0 * (1.0 + p3 * temp)
    
    # Yield stress: Y(T) = p1 * (1 + p4 * temp)
    Y = p1 * (1.0 + p4 * temp)
    
    # Yield strain: eps_y(T) = p2 * (1 + p5 * temp)
    eps_y = p2 * (1.0 + p5 * temp)
    
    # Hardening modulus: H(T) = p6 * (1 + p7 * temp)
    H = p6 * (1.0 + p7 * temp)
    
    # Smooth transition width
    w = np.abs(p8) + 0.01  # ensure positive width
    
    # Smoothstep function for transition
    # x = (strain - eps_y) / w
    x = (strain - eps_y) / w
    
    # Smoothstep: 3*x^2 - 2*x^3 for x in [0,1], 0 for x<0, 1 for x>1
    x_clipped = np.clip(x, 0.0, 1.0)
    smooth = 3.0 * x_clipped**2 - 2.0 * x_clipped**3
    
    # Elastic stress
    stress_elastic = E * strain
    
    # Plastic stress
    stress_plastic = Y + H * np.maximum(strain - eps_y, 0.0)
    
    # Blend between elastic and plastic
    stress = (1.0 - smooth) * stress_elastic + smooth * stress_plastic
    
    # Higher-order correction terms
    correction = p9 * strain**2 * (1.0 - smooth)
    
    return stress + correction
