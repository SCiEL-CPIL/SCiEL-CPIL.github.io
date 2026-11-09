import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation of motion for a nonlinear oscillator.
    Model: a = F_drive(t) + F_restoring(x) + F_damping(v) + CrossTerms
    Assumes fundamental frequency of 1 for the drive.
    """
    # Drive terms (Fourier basis with omega=1)
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Position terms (Polynomial basis)
    x2 = x * x
    x3 = x2 * x
    
    # Velocity terms (Polynomial basis)
    v2 = v * v
    v3 = v2 * v
    
    # Cross term
    xv = x * v
    
    # Combine
    # params:
    # 0: sin(t) amp
    # 1: cos(t) amp
    # 2: x coeff
    # 3: x^2 coeff
    # 4: x^3 coeff
    # 5: v coeff
    # 6: v^2 coeff
    # 7: v^3 coeff
    # 8: xv coeff
    # 9: bias
    
    a = (params[0] * sin_t + 
         params[1] * cos_t + 
         params[2] * x + 
         params[3] * x2 + 
         params[4] * x3 + 
         params[5] * v + 
         params[6] * v2 + 
         params[7] * v3 + 
         params[8] * xv + 
         params[9])
         
    return a
