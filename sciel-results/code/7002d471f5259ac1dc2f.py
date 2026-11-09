import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to O(1) to improve optimization conditioning
    S_x = 0.5728
    S_v = 0.2821
    X = x / S_x
    V = v / S_v
    
    # Chebyshev basis of degree 3 in X and V
    # T_0=1, T_1=X, T_2=2X^2-1, T_3=4X^3-3X
    # S_0=1, S_1=V, S_2=2V^2-1, S_3=4V^3-3V
    # 10 terms: products T_i * S_j for i+j <= 3
    # (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), (3,0), (2,1), (1,2), (0,3)
    
    T0 = 1.0
    T1 = X
    T2 = 2.0 * X**2 - 1.0
    T3 = 4.0 * X**3 - 3.0 * X
    
    S0 = 1.0
    S1 = V
    S2 = 2.0 * V**2 - 1.0
    S3 = 4.0 * V**3 - 3.0 * V
    
    return (params[0] * T0 * S0 + 
            params[1] * T1 * S0 + 
            params[2] * T0 * S1 + 
            params[3] * T2 * S0 + 
            params[4] * T1 * S1 + 
            params[5] * T0 * S2 + 
            params[6] * T3 * S0 + 
            params[7] * T2 * S1 + 
            params[8] * T1 * S2 + 
            params[9] * T0 * S3)
