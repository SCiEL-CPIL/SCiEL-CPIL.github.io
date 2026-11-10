import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    # Scale x to [-1, 1] to improve conditioning of polynomial/trig terms
    u = x / 2.0
    
    # Pre-compute basis functions
    # Chebyshev polynomials T_0 to T_4
    T0 = np.ones_like(u)
    T1 = u
    T2 = 2 * u * u - 1
    T3 = 4 * u * u * u - 3 * u
    T4 = 8 * u * u * u * u - 8 * u * u + 1
    
    # Trigonometric
    sin_term = np.sin(np.pi * u)
    cos_term = np.cos(np.pi * u)
    
    # Gaussian decay
    exp_term = np.exp(-2.0 * u * u)
    
    # Saturating non-linearity
    tanh_term = np.tanh(u)
    
    # Odd Gaussian
    odd_gauss = u * np.exp(-u * u)
    
    # Linear combination
    # p[0]..p[9]
    pred = (params[0] * T0 + 
            params[1] * T1 + 
            params[2] * T2 + 
            params[3] * T3 + 
            params[4] * T4 + 
            params[5] * sin_term + 
            params[6] * cos_term + 
            params[7] * exp_term + 
            params[8] * tanh_term + 
            params[9] * odd_gauss)
            
    return pred
