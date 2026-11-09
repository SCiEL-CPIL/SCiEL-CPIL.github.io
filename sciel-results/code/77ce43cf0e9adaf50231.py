"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a Generalized Additive Model (GAM).
    
    Inputs are assumed to be in the following ranges based on training data:
    - b: [1, 5]
    - s: [1, 5]
    - temp: [1, 40]
    - pH: [2, 12]
    
    We use scaled polynomial basis functions for each variable to ensure
    numerical stability and convexity for the BFGS optimizer.
    
    Parameters:
    - params[0]: Bias
    - params[1], params[2]: Linear, Quadratic basis for b
    - params[3], params[4]: Linear, Quadratic basis for s
    - params[5], params[6]: Linear, Quadratic basis for temp
    - params[7], params[8]: Linear, Quadratic basis for pH
    - params[9]: Cubic basis for temp (temperature sensitivity)
    """
    # Scale inputs to approximately [-1, 1] or similar centered range
    # b: center 3, scale 2
    b_scaled = (b - 3.0) / 2.0
    # s: center 3, scale 2
    s_scaled = (s - 3.0) / 2.0
    # temp: center 20, scale 10
    temp_scaled = (temp - 20.0) / 10.0
    # pH: center 7, scale 5
    ph_scaled = (pH - 7.0) / 5.0
    
    # Basis functions
    # Use Gaussian bumps for Temperature and pH to capture non-monotonic optima
    # Use polynomial terms for b and s to capture saturation/curvature
    
    # Temperature Bumps (centers in scaled space: -1.0, 0.5, 1.7)
    # Width 0.8
    t_center_1 = -1.0
    t_center_2 = 0.5
    t_center_3 = 1.7
    t_width = 0.8
    
    temp_bump_1 = np.exp(-((temp_scaled - t_center_1) / t_width) ** 2)
    temp_bump_2 = np.exp(-((temp_scaled - t_center_2) / t_width) ** 2)
    temp_bump_3 = np.exp(-((temp_scaled - t_center_3) / t_width) ** 2)
    
    # pH Bumps (centers in scaled space: -0.6, 0.0)
    # Width 0.8
    ph_center_1 = -0.6
    ph_center_2 = 0.0
    ph_width = 0.8
    
    ph_bump_1 = np.exp(-((ph_scaled - ph_center_1) / ph_width) ** 2)
    ph_bump_2 = np.exp(-((ph_scaled - ph_center_2) / ph_width) ** 2)
    
    # Polynomial bases for b and s
    b_lin = b_scaled
    b_quad = b_scaled ** 2
    
    s_lin = s_scaled
    s_quad = s_scaled ** 2
    
    # Combine linearly
    # params[0]: Bias
    # params[1-3]: Temp bumps
    # params[4-5]: pH bumps
    # params[6-7]: b lin, quad
    # params[8-9]: s lin, quad
    
    pred = (params[0] + 
            params[1] * temp_bump_1 + 
            params[2] * temp_bump_2 + 
            params[3] * temp_bump_3 +
            params[4] * ph_bump_1 + 
            params[5] * ph_bump_2 +
            params[6] * b_lin + params[7] * b_quad +
            params[8] * s_lin + params[9] * s_quad)
            
    return pred
# EVOLVE-BLOCK-END
