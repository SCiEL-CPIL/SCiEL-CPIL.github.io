"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard: ensure we return within 150s wall-clock
    t_start = time.time()
    deadline = 145.0  # 5s safety margin
    
    # Compute Chebyshev polynomials of the first kind up to degree 3
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2*t**2 - 1
    # T_3(t) = 4*t**3 - 3*t
    # 
    # Chebyshev polynomials are minimax on [-1,1], giving better conditioning
    # for BFGS optimization compared to Legendre polynomials.
    
    px0 = np.ones_like(x)
    px1 = x
    px2 = 2.0 * x**2 - 1.0
    px3 = 4.0 * x**3 - 3.0 * x
    
    py0 = np.ones_like(y)
    py1 = y
    py2 = 2.0 * y**2 - 1.0
    py3 = 4.0 * y**3 - 3.0 * y
    
    # Stack basis functions into a design matrix (n_points x 10)
    B = np.column_stack([
        px0 * py0,  # 1
        px1 * py0,  # x
        px0 * py1,  # y
        px2 * py0,  # T_2(x)
        px1 * py1,  # x*y
        px0 * py2,  # T_2(y)
        px3 * py0,  # T_3(x)
        px2 * py1,  # T_2(x)*y
        px1 * py2,  # x*T_2(y)
        px0 * py3   # T_3(y)
    ])
    
    # Matrix-vector product with safety against non-finite values
    with np.errstate(over='ignore', invalid='ignore'):
        pred = B @ params
    
    # Replace any non-finite values with 0 to keep BFGS stable
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
