import numpy as np
import time

# Deadline guard: 150s wall-clock limit, use 145s to be safe
_DEADLINE = 145.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 1 input.
    
    Strategy: Use a hybrid basis that combines:
    1. Chebyshev polynomials (degrees 0-5) for smooth global structure
    2. Gaussian RBFs (centers at -1.5, -0.5, 0.5, 1.5 with width 1.0) for local features
    
    The model is linear in params, so BFGS will find the global optimum efficiently.
    Input is scaled to [-1, 1] for numerical stability.
    """
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params is a flat array of length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
    
    try:
        # Scale x from [-2, 2] to [-1, 1] for better conditioning
        xs = x / 2.0
        
        n = xs.size
        
        # Chebyshev polynomials T_0 through T_5
        # T_0 = 1
        # T_1 = x
        # T_2 = 2x^2 - 1
        # T_3 = 4x^3 - 3x
        # T_4 = 8x^4 - 8x^2 + 1
        # T_5 = 16x^5 - 20x^3 + 5x
        c0 = np.ones_like(xs)
        c1 = xs.copy()
        c2 = 2.0 * xs**2 - 1.0
        c3 = 4.0 * xs**3 - 3.0 * xs
        c4 = 8.0 * xs**4 - 8.0 * xs**2 + 1.0
        c5 = 16.0 * xs**5 - 20.0 * xs**3 + 5.0 * xs
        
        # RBF basis with wider spread
        centers = np.array([-1.5, -0.5, 0.5, 1.5])
        width = 1.0
        sig2 = 2.0 * width**2
        r0 = np.exp(-(x - centers[0])**2 / sig2)
        r1 = np.exp(-(x - centers[1])**2 / sig2)
        r2 = np.exp(-(x - centers[2])**2 / sig2)
        r3 = np.exp(-(x - centers[3])**2 / sig2)
        
        # Stack into basis matrix (n x 10)
        basis = np.column_stack([c0, c1, c2, c3, c4, c5, r0, r1, r2, r3])
        
        # Linear model: y = basis @ params
        return basis @ params
        
    except Exception:
        return np.zeros_like(x)
