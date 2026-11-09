import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Scale x to [-1, 1] for better conditioning of basis functions
    t = x / 2.0
    
    # Chebyshev polynomials of the first kind T_n(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    # Fourier terms (sine/cosine) to capture periodicity
    # Using pi * t so that at x=2 (t=1) we have sin(pi), cos(pi), etc.
    S1 = np.sin(np.pi * t)
    C1 = np.cos(np.pi * t)
    S2 = np.sin(2.0 * np.pi * t)
    C2 = np.cos(2.0 * np.pi * t)
    
    # Gaussian term at center 0 (t=0)
    G0 = np.exp(-t**2)
    
    # Combine
    y_hat = (p[0] * T0 + 
             p[1] * T1 + 
             p[2] * T2 + 
             p[3] * T3 + 
             p[4] * T4 + 
             p[5] * S1 + 
             p[6] * C1 + 
             p[7] * S2 + 
             p[8] * C2 + 
             p[9] * G0)
             
    return y_hat
