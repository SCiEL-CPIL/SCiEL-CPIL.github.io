"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a hybrid linear-tanh basis.
    
    This approach combines a global linear trend with a single non-linear 
    hyperbolic tangent term. The tanh term allows the model to capture 
    saturation or threshold effects (common in biological growth) while 
    the linear term fits the overall slope. Inputs are normalized to 
    improve the conditioning of the optimization problem.
    
    Parameters:
    b: Population density
    s: Substrate concentration
    temp: Temperature
    pH: pH level
    params: Array of 10 fitted constants.
        p[0]: Weight for b in tanh term
        p[1]: Weight for s in tanh term
        p[2]: Weight for temp in tanh term
        p[3]: Weight for pH in tanh term
        p[4]: Bias for tanh term
        p[5]: Scale factor for tanh term
        p[6]: Weight for b in linear term
        p[7]: Weight for s in linear term
        p[8]: Weight for temp in linear term
        p[9]: Weight for pH in linear term
    """
    p = params
    
    # Normalize inputs to roughly [0, 1] or [-1, 1] range for better conditioning.
    # Ranges: b:[1,5], s:[1,5], temp:[1,40], pH:[2,12]
    b_norm = b / 5.0
    s_norm = s / 5.0
    temp_norm = temp / 40.0
    pH_norm = pH / 12.0
    
    # Non-linear part: tanh(w . x + b)
    z = (p[0] * b_norm + 
         p[1] * s_norm + 
         p[2] * temp_norm + 
         p[3] * pH_norm + 
         p[4])
         
    non_linear_part = p[5] * np.tanh(z)
    
    # Linear part
    linear_part = (p[6] * b_norm + 
                   p[7] * s_norm + 
                   p[8] * temp_norm + 
                   p[9] * pH_norm)
                   
    return non_linear_part + linear_part
# EVOLVE-BLOCK-END
