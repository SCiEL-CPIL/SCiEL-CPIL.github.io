import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_4
    T = []
    T.append(np.ones_like(t))  # T_0
    T.append(t.copy())          # T_1
    for k in range(2, 5):
        T_curr = 2.0 * t * T[-1] - T[-2]
        T.append(T_curr)
    
    # Bilinear basis: T_i * T_j for selected pairs
    # This allows capturing sharp features like |x|, x^3 more efficiently
    # Pairs: (0,0), (0,1), (0,2), (0,3), (0,4), (1,1), (1,2), (1,3), (2,2), (2,3)
    # That's 10 basis functions, linear in params
    
    y = np.zeros_like(t)
    y += params[0] * T[0] * T[0]  # T_0 * T_0 = 1
    y += params[1] * T[0] * T[1]  # T_0 * T_1 = t
    y += params[2] * T[0] * T[2]  # T_0 * T_2
    y += params[3] * T[0] * T[3]  # T_0 * T_3
    y += params[4] * T[0] * T[4]  # T_0 * T_4
    y += params[5] * T[1] * T[1]  # T_1 * T_1 = t^2
    y += params[6] * T[1] * T[2]  # T_1 * T_2
    y += params[7] * T[1] * T[3]  # T_1 * T_3
    y += params[8] * T[2] * T[2]  # T_2 * T_2
    y += params[9] * T[2] * T[3]  # T_2 * T_3
    
    return y
