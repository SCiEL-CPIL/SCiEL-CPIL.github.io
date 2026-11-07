import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters.
    Basis includes constant, linear terms, and sine/cosine terms for each variable.
    This allows fitting functions with periodic components.
    
    Basis (10 terms):
    0: 1
    1: x
    2: y
    3: z
    4: sin(x)
    5: cos(x)
    6: sin(y)
    7: cos(y)
    8: sin(z)
    9: cos(z)
    """
    p = params
    return (
        p[0] * 1.0 +
        p[1] * x +
        p[2] * y +
        p[3] * z +
        p[4] * np.sin(x) +
        p[5] * np.cos(x) +
        p[6] * np.sin(y) +
        p[7] * np.cos(y) +
        p[8] * np.sin(z) +
        p[9] * np.cos(z)
    )
