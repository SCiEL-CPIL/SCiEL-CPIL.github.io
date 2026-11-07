"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a truncated Fourier series basis.
    This approach uses sine and cosine terms to approximate the target function.
    The domain [-3, 3] is mapped to [-pi, pi] via scaling factor omega = pi/3.
    
    params: array of length 10.
    c0: DC term
    c1, c2: 1st harmonic (sin, cos)
    c3, c4: 2nd harmonic (sin, cos)
    c5, c6: 3rd harmonic (sin, cos)
    c7, c8: 4th harmonic (sin, cos)
    c9: 5th harmonic (sin only, to fit 10 params)
    """
    x = np.asarray(x)
    n = len(x)
    
    # Scaling to map [-3, 3] to [-pi, pi]
    omega = np.pi / 3.0
    theta = omega * x
    
    # Initialize feature matrix
    # 10 features
    F = np.zeros((n, 10))
    
    if n == 0:
        return np.array([])
        
    # c0: DC term
    F[:, 0] = 1.0
    
    # Harmonics 1 to 4 (full sin/cos pairs)
    # k=1: indices 1, 2
    # k=2: indices 3, 4
    # k=3: indices 5, 6
    # k=4: indices 7, 8
    
    F[:, 1] = np.sin(1 * theta)
    F[:, 2] = np.cos(1 * theta)
    
    F[:, 3] = np.sin(2 * theta)
    F[:, 4] = np.cos(2 * theta)
    
    F[:, 5] = np.sin(3 * theta)
    F[:, 6] = np.cos(3 * theta)
    
    F[:, 7] = np.sin(4 * theta)
    F[:, 8] = np.cos(4 * theta)
    
    # 5th harmonic: only sin to use the 10th parameter
    F[:, 9] = np.sin(5 * theta)
    
    # Prediction
    pred = F @ params
    
    return pred
# EVOLVE-BLOCK-END
