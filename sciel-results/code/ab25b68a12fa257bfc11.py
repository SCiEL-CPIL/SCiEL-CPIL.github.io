import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a Quadratic Response Surface.
    
    This approach uses a polynomial basis on normalized inputs.
    It is structurally different from the multiplicative biological model,
    offering a smoother loss landscape for the BFGS optimizer.
    
    Inputs are normalized to roughly [-1, 1] to improve conditioning.
    b: [1, 5] -> center 3, scale 2
    s: [1, 5] -> center 3, scale 2
    temp: [1, 40] -> center 20.5, scale 19.5
    pH: [2, 12] -> center 7, scale 5
    
    Model:
    y = p0 + p1*b_n + p2*s_n + p3*T_n + p4*pH_n 
        + p5*b_n^2 + p6*s_n^2 + p7*T_n^2 + p8*pH_n^2 
        + p9*T_n*pH_n
        
    The last term p9 allows for interaction between temperature and pH,
    which is biologically plausible. We drop other interactions to stay within 10 params.
    """
    p = params
    
    # Normalize inputs to improve numerical stability for BFGS
    b_n = (b - 3.0) / 2.0
    s_n = (s - 3.0) / 2.0
    T_n = (temp - 20.5) / 19.5
    pH_n = (pH - 7.0) / 5.0
    
    # Quadratic terms
    b2 = b_n * b_n
    s2 = s_n * s_n
    T2 = T_n * T_n
    pH2 = pH_n * pH_n
    
    # Interaction term (Temp and pH are often coupled in biological systems)
    T_pH = T_n * pH_n
    
    # Linear combination
    # p0: intercept
    # p1-p4: linear effects
    # p5-p8: quadratic effects (curvature)
    # p9: Temp-pH interaction
    
    mu = (p[0] 
          + p[1] * b_n 
          + p[2] * s_n 
          + p[3] * T_n 
          + p[4] * pH_n 
          + p[5] * b2 
          + p[6] * s2 
          + p[7] * T2 
          + p[8] * pH2 
          + p[9] * T_pH)
          
    return mu
