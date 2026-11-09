"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns prediction using a Fourier series basis.
    The domain is [-3, 3], so we use a period of 6.
    Basis functions:
    0: 1
    1: cos(pi*x/3)
    2: sin(pi*x/3)
    3: cos(2*pi*x/3)
    4: sin(2*pi*x/3)
    5: cos(3*pi*x/3) = cos(pi*x)
    6: sin(3*pi*x/3) = sin(pi*x)
    7: cos(4*pi*x/3)
    8: sin(4*pi*x/3)
    9: cos(5*pi*x/3)
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.flatten()
        
    # Pre-compute arguments
    # Fundamental frequency w = pi / 3
    w = np.pi / 3.0
    arg1 = w * x
    arg2 = 2 * w * x
    arg3 = 3 * w * x
    arg4 = 4 * w * x
    arg5 = 5 * w * x
    
    # Construct basis matrix
    # Shape: (N, 10)
    basis = np.empty((x.size, 10), dtype=np.float64)
    basis[:, 0] = 1.0
    basis[:, 1] = np.cos(arg1)
    basis[:, 2] = np.sin(arg1)
    basis[:, 3] = np.cos(arg2)
    basis[:, 4] = np.sin(arg2)
    basis[:, 5] = np.cos(arg3)
    basis[:, 6] = np.sin(arg3)
    basis[:, 7] = np.cos(arg4)
    basis[:, 8] = np.sin(arg4)
    basis[:, 9] = np.cos(arg5)
    
    pred = basis @ params
    return pred
# EVOLVE-BLOCK-END
