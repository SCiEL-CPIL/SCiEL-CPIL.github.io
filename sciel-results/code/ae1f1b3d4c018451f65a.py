import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale inputs to [-1, 1] to utilize Chebyshev polynomials optimally
    # x range: [-0.5728, 0.5275] -> max abs ~ 0.5728
    # v range: [-0.1754, 0.2821] -> max abs ~ 0.2821
    scale_x = 0.5728
    scale_v = 0.2821
    
    X = x / scale_x
    V = v / scale_v
    
    # Clip to [-1, 1] to prevent numerical issues if slightly out of range
    # though training is in range. Chebyshev grows outside [-1,1].
    # It's safer to allow growth if the model needs it, but for stability,
    # we assume in-distribution.
    
    # Chebyshev polynomials of the first kind
    T0_X = 1.0
    T1_X = X
    T2_X = 2.0 * X * X - 1.0
    T3_X = 4.0 * X * X * X - 3.0 * X
    
    T0_V = 1.0
    T1_V = V
    T2_V = 2.0 * V * V - 1.0
    T3_V = 4.0 * V * V * V - 3.0 * V
    
    # Construct 10 basis terms mixing X and V Chebyshev polynomials
    # Focus on low-order terms first for stability, then cross terms.
    
    term1 = T0_X * T0_V       # 1
    term2 = T1_X * T0_V       # X
    term3 = T0_X * T1_V       # V
    term4 = T2_X * T0_V       # 2X^2 - 1
    term5 = T0_X * T2_V       # 2V^2 - 1
    term6 = T1_X * T1_V       # X*V
    term7 = T3_X * T0_V       # 4X^3 - 3X
    term8 = T0_X * T3_V       # 4V^3 - 3V
    term9 = T2_X * T1_V       # (2X^2 - 1)*V
    term10 = T1_X * T2_V      # X*(2V^2 - 1)
    
    # Linear combination
    return (params[0] * term1 +
            params[1] * term2 +
            params[2] * term3 +
            params[3] * term4 +
            params[4] * term5 +
            params[5] * term6 +
            params[6] * term7 +
            params[7] * term8 +
            params[8] * term9 +
            params[9] * term10)
