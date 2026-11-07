import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 5] -> [0, 1]
    # s: [1, 5] -> [0, 1]
    # temp: [1, 40] -> [0, 1]
    # pH: [2, 12] -> [0, 1]
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # Construct 10 basis functions for a sparse 4th-order multivariate polynomial
    # We select terms that capture linear effects, quadratic curvature (for optima),
    # and key interaction terms.
    
    # 1. Constant
    phi_0 = np.ones_like(b_norm)
    
    # 2-5. Linear terms
    phi_1 = b_norm
    phi_2 = s_norm
    phi_3 = t_norm
    phi_4 = ph_norm
    
    # 6-8. Quadratic terms (curvature for temperature, substrate, pH)
    phi_5 = t_norm * t_norm
    phi_6 = s_norm * s_norm
    phi_7 = ph_norm * ph_norm
    
    # 9-10. Interaction terms (Temperature-Substrate, Temperature-pH)
    phi_8 = t_norm * s_norm
    phi_9 = t_norm * ph_norm
    
    # Linear combination of basis functions
    y = (p[0] * phi_0 +
         p[1] * phi_1 +
         p[2] * phi_2 +
         p[3] * phi_3 +
         p[4] * phi_4 +
         p[5] * phi_5 +
         p[6] * phi_6 +
         p[7] * phi_7 +
         p[8] * phi_8 +
         p[9] * phi_9)
         
    return y
