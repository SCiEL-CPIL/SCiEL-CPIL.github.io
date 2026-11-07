"""Program using a richer global basis (polynomial + trig + smooth) for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form symbolic regression candidate using a mixed basis:
    - Scaled polynomial terms (degrees 0-3) for smooth global structure
    - Sine/cosine terms for periodic components
    - tanh and arctan for smooth nonlinear transitions
    
    10 parameters are used as weights for the 10 basis functions.
    Input is scaled to [-1, 1] for numerical stability.
    """
    # Deadline guard
    t0 = time.time()
    
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Scale input to [-1, 1]
    xs = x / 3.0
    
    # Build basis matrix (n_points, 10)
    # Polynomial terms: 1, xs, xs^2, xs^3
    # Trig terms: sin(pi*xs), cos(pi*xs), sin(2*pi*xs), cos(2*pi*xs)
    # Smooth nonlinear: tanh(xs), arctan(xs)
    
    basis = np.empty((x.shape[0], 10), dtype=np.float64)
    basis[:, 0] = 1.0
    basis[:, 1] = xs
    basis[:, 2] = xs ** 2
    basis[:, 3] = xs ** 3
    basis[:, 4] = np.sin(np.pi * xs)
    basis[:, 5] = np.cos(np.pi * xs)
    basis[:, 6] = np.sin(2.0 * np.pi * xs)
    basis[:, 7] = np.cos(2.0 * np.pi * xs)
    basis[:, 8] = np.tanh(xs)
    basis[:, 9] = np.arctan(xs)
    
    # Linear combination
    y = basis @ params
    
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
# EVOLVE-BLOCK-END
