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
    
    # Sparse 4th-order multivariate polynomial
    # Terms:
    # p[0]: Constant
    # p[1]*b, p[2]*s, p[3]*t, p[4]*ph : Linear terms
    # p[5]*t^2, p[6]*ph^2, p[7]*s^2 : Quadratic terms for optima/saturation
    # p[8]*t*ph : Interaction between temperature and pH
    # p[9]*t*s : Interaction between temperature and substrate
    
    y = (p[0] 
         + p[1] * b_norm 
         + p[2] * s_norm 
         + p[3] * t_norm 
         + p[4] * ph_norm 
         + p[5] * t_norm**2 
         + p[6] * ph_norm**2 
         + p[7] * s_norm**2 
         + p[8] * t_norm * ph_norm 
         + p[9] * t_norm * s_norm)
         
    return y
