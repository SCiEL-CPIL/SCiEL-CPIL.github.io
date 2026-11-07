import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x is expected to be in [-2, 2]
    
    # Basis functions
    # 1. Constant
    b0 = np.ones_like(x)
    # 2. Linear
    b1 = x
    # 3. Quadratic
    b2 = x**2
    # 4. Cubic
    b3 = x**3
    # 5. Quartic
    b4 = x**4
    # 6. Sine (period 2*pi ~ 6.28)
    b5 = np.sin(x)
    # 7. Cosine
    b6 = np.cos(x)
    # 8. Sine 2x (period pi ~ 3.14)
    b7 = np.sin(2 * x)
    # 9. Cosine 2x
    b8 = np.cos(2 * x)
    # 10. Gaussian bump centered at 0
    b9 = np.exp(-0.5 * x**2)
    
    # Combine
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
