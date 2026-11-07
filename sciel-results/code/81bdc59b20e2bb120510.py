import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-inspired basis for nonlinear oscillator.
    Includes polynomial terms up to 5th order in x and v, and cross terms.
    No dynamic normalization to avoid overhead and potential instability.
    """
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    
    v2 = v * v
    v3 = v2 * v
    
    T = np.column_stack([
        x,
        x3,
        x5,
        v,
        v3,
        x2 * v,
        x * v2,
        x * v,
        x2,
        np.ones_like(x)
    ])
    
    return T @ params
