import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a spectral (Fourier) basis.
    The basis functions are sin/cos of linear combinations of x and v.
    This structure is better conditioned for BFGS optimization as derivatives are bounded.
    
    Structure:
    - p[0] * sin(p[1]*x + p[2]*v)
    - p[3] * cos(p[4]*x + p[5]*v)
    - p[6] * sin(p[7]*x - p[8]*v)
    - p[9] * cos(x)  # Fixed frequency 1.0 for stability, or maybe p[9] is amplitude of cos(1.0*x)
    
    Wait, I have 10 params. Let's distribute them:
    Term 1: A1 * sin(k1_x * x + k1_v * v)  -> p[0]=A1, p[1]=k1_x, p[2]=k1_v
    Term 2: A2 * cos(k2_x * x + k2_v * v)  -> p[3]=A2, p[4]=k2_x, p[5]=k2_v
    Term 3: A3 * sin(k3_x * x + k3_v * v)  -> p[6]=A3, p[7]=k3_x, p[8]=k3_v
    Term 4: A4 * cos(x)                     -> p[9]=A4 (frequency fixed at 1.0 for simplicity/stability)
    
    This uses all 10 parameters.
    """
    p = params
    
    # Compute arguments
    arg1 = p[1] * x + p[2] * v
    arg2 = p[4] * x + p[5] * v
    arg3 = p[7] * x + p[8] * v
    
    # Compute terms
    term1 = p[0] * np.sin(arg1)
    term2 = p[3] * np.cos(arg2)
    term3 = p[6] * np.sin(arg3)
    term4 = p[9] * np.cos(x)
    
    return term1 + term2 + term3 + term4
