import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a physics-informed basis of bounded nonlinearities.
    
    The basis is designed to be well-conditioned for BFGS by avoiding near-collinear terms
    (e.g., x and sin(x) are removed/replaced with higher-order bounded terms).
    
    Restoring forces (odd in x):
    - sin(x)
    - sin(x) * x^2 (approximates x^3, bounded)
    - sin(2*x) (captures harmonics)
    
    Damping forces (odd in v):
    - sin(v)
    - sin(v) * v^2 (approximates v^3, bounded)
    - sin(2*v) (captures harmonics)
    
    Coupling and Bias:
    - x * v
    - sin(x) * sin(v)
    - tanh(x) * x (another odd term, distinct from sin(x) slightly)
    - Bias
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Precompute basis functions
    sin_x = np.sin(x)
    sin_2x = np.sin(2 * x)
    sin_x_x2 = sin_x * x * x
    
    sin_v = np.sin(v)
    sin_2v = np.sin(2 * v)
    sin_v_v2 = sin_v * v * v
    
    tanh_x = np.tanh(x)
    tanh_x_x = tanh_x * x
    
    xv = x * v
    sin_x_sin_v = sin_x * sin_v

    # Linear combination of basis functions
    return (
        p0 * sin_x
        + p1 * sin_x_x2
        + p2 * sin_2x
        + p3 * sin_v
        + p4 * sin_v_v2
        + p5 * sin_2v
        + p6 * xv
        + p7 * sin_x_sin_v
        + p8 * tanh_x_x
        + p9
    )
