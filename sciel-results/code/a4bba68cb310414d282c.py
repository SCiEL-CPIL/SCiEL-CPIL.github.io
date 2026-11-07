"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod.
    
    Uses a linear-in-parameters basis function approach for robust BFGS optimization.
    The basis functions capture:
    - Elastic linear region (strain)
    - Plastic hardening (quadratic/cubic strain terms)
    - Temperature dependence (linear temp, interactions)
    - Non-linear strain effects (higher order terms)
    
    This linear structure ensures the MSE loss is convex in params,
    allowing BFGS to reliably find the global minimum.
    """
    s = strain
    t = temp
    p = params
    
    # Normalize inputs to improve numerical conditioning
    # Training ranges: strain [-0.001911, 0.9636], temp [0.06667, 1.0]
    s_min = -0.001911
    s_max = 0.9636
    t_min = 0.06667
    t_max = 1.0
    
    s_norm = (s - s_min) / (s_max - s_min)
    t_norm = (t - t_min) / (t_max - t_min)
    
    # Basis functions
    # 1. Constant
    b0 = np.ones_like(s)
    
    # 2. Linear strain
    b1 = s_norm
    
    # 3. Quadratic strain
    b2 = s_norm ** 2
    
    # 4. Cubic strain
    b3 = s_norm ** 3
    
    # 5. Linear temperature
    b4 = t_norm
    
    # 6. Strain-temperature interaction (linear)
    b5 = s_norm * t_norm
    
    # 7. Quadratic strain-temperature interaction
    b6 = s_norm ** 2 * t_norm
    
    # 8. Temperature squared
    b7 = t_norm ** 2
    
    # 9. Strain cubed - temperature interaction
    b8 = s_norm ** 3 * t_norm
    
    # 10. Pure temperature effect on yield
    b9 = np.where(s_norm > 0.01, 1.0, 0.0) * t_norm
    
    # Combine basis functions with parameters
    pred = (p[0] * b0 + 
            p[1] * b1 + 
            p[2] * b2 + 
            p[3] * b3 + 
            p[4] * b4 + 
            p[5] * b5 + 
            p[6] * b6 + 
            p[7] * b7 + 
            p[8] * b8 + 
            p[9] * b9)
    
    return pred
# EVOLVE-BLOCK-END
