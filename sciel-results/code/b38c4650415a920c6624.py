"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function f(x, y) using a linear combination of Chebyshev polynomials.
    The basis is chosen to be orthogonal over [-1, 1] x [-1, 1] to aid BFGS convergence.
    
    Basis functions (10 total):
    1. 1
    2. T1(x) = x
    3. T2(x) = 2x^2 - 1
    4. T1(y) = y
    5. T2(y) = 2y^2 - 1
    6. T1(x)T1(y) = xy
    7. T1(x)T2(y) = x(2y^2 - 1)
    8. T2(x)T1(y) = (2x^2 - 1)y
    9. T2(x)T2(y) = (2x^2 - 1)(2y^2 - 1)
    10. T3(x) = 4x^3 - 3x
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Strategy: Product-of-sines architecture
    # Model: p0 * sin(p1*x + p2*y + p3) + p4
    # This allows BFGS to adjust amplitude, frequencies, and phase.
    # We use the first 5 parameters. The rest are ignored.
    # To ensure stability, we might want to scale the arguments to sine
    # but BFGS should handle the scaling of p1, p2.
    
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    
    # Calculate the argument
    arg = p1 * x + p2 * y + p3
    
    # Calculate sine
    # np.sin is safe for finite inputs
    sin_val = np.sin(arg)
    
    # Combine
    pred = p0 * sin_val + p4
    
    return pred
# EVOLVE-BLOCK-END
