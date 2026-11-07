# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 115.0  # Safety margin within 120s limit


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid fixed-basis model using Chebyshev polynomials, Gaussian RBFs, and trigonometric functions.
    
    This approach combines the stability of a linear least-squares problem (ideal for BFGS)
    with a basis set capable of representing smooth, polynomial, periodic, and localized functions.
    
    Parameters:
    - p[0:5] : Coefficients for Chebyshev polynomials T_0 to T_4 (mapped to [-1, 1])
    - p[5:10]: Coefficients for Gaussian RBFs at fixed centers {-3, -1.5, 0, 1.5, 3}
    
    Note: We prioritize RBFs over higher-order Chebyshev or trig terms because the observed
    stagnation suggests the target may have localized features or sharp transitions that
    global polynomials/trigs struggle with, while RBFs provide local flexibility.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Safety check: if we are running out of time, return a simple fallback
    # However, since this is a single forward pass, it should be fast.
    # The deadline guard is mainly to prevent pathological cases if the evaluator
    # calls this in a tight loop or if numpy operations are unexpectedly slow.
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback: simple linear interpolation approximation
        return params[0] + params[1] * x

    # Normalize x to [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Calculate Chebyshev Polynomials T_0 to T_4
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    
    b0 = np.ones_like(x)
    b1 = t
    b2 = 2.0 * t * t - 1.0
    b3 = 4.0 * b1 * b1 * b1 - 3.0 * b1
    b4 = 8.0 * b1 * b1 * b1 * b1 - 8.0 * b1 * b1 + 1.0
    
    # Gaussian RBFs at fixed centers
    # Centers chosen to cover the domain [-3, 3]
    # Width chosen to be moderate (1.0) to allow overlap but maintain locality
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    width = 1.0
    
    # Compute RBF values: exp(-0.5 * ((x - center) / width)^2)
    # We compute each RBF separately for clarity and to avoid large intermediate arrays if needed,
    # though vectorization is possible. Given N=200, this is negligible.
    
    b5 = np.exp(-0.5 * ((x - centers[0]) / width) ** 2)
    b6 = np.exp(-0.5 * ((x - centers[1]) / width) ** 2)
    b7 = np.exp(-0.5 * ((x - centers[2]) / width) ** 2)
    b8 = np.exp(-0.5 * ((x - centers[3]) / width) ** 2)
    b9 = np.exp(-0.5 * ((x - centers[4]) / width) ** 2)
    
    # Linear combination using the 10 parameters
    y = (params[0] * b0 + 
         params[1] * b1 + 
         params[2] * b2 + 
         params[3] * b3 + 
         params[4] * b4 + 
         params[5] * b5 + 
         params[6] * b6 + 
         params[7] * b7 + 
         params[8] * b8 + 
         params[9] * b9)
         
    return y
# EVOLVE-BLOCK-END
