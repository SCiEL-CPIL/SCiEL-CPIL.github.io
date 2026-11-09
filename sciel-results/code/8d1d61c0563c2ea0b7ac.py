import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Compute Fourier basis functions on [-1, 1]
    # Period is 2, so fundamental frequency is pi
    
    # First harmonics
    cos_pi_x = np.cos(np.pi * x)
    sin_pi_x = np.sin(np.pi * x)
    cos_pi_y = np.cos(np.pi * y)
    sin_pi_y = np.sin(np.pi * y)
    
    # Second harmonics
    cos_2pi_x = np.cos(2 * np.pi * x)
    sin_2pi_x = np.sin(2 * np.pi * x)
    cos_2pi_y = np.cos(2 * np.pi * y)
    sin_2pi_y = np.sin(2 * np.pi * y)
    
    # Basis functions - complete tensor-product Fourier basis
    # b0: Constant
    b0 = np.ones_like(x)
    # b1: Cos(πx)
    b1 = cos_pi_x
    # b2: Sin(πx)
    b2 = sin_pi_x
    # b3: Cos(πy)
    b3 = cos_pi_y
    # b4: Sin(πy)
    b4 = sin_pi_y
    # b5: Interaction Cos(πx)Cos(πy)
    b5 = cos_pi_x * cos_pi_y
    # b6: Interaction Sin(πx)Sin(πy)
    b6 = sin_pi_x * sin_pi_y
    # b7: Second harmonic x - Cos(2πx)
    b7 = cos_2pi_x
    # b8: Second harmonic y - Sin(2πy)
    b8 = sin_2pi_y
    # b9: Mixed interaction Cos(πx)Sin(πy)
    b9 = cos_pi_x * sin_pi_y
    
    return (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
