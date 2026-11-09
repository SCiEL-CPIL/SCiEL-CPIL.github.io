"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a well-conditioned basis of 
    tensor-product Chebyshev polynomials up to total degree 3.
    
    The basis functions are:
    T_0(x)T_0(y), T_1(x)T_0(y), T_0(x)T_1(y),
    T_2(x)T_0(y), T_1(x)T_1(y), T_0(x)T_2(y),
    T_3(x)T_0(y), T_2(x)T_1(y), T_1(x)T_2(y), T_0(x)T_3(y)
    
    Chebyshev polynomials have superior approximation properties for smooth
    non-polynomial functions on [-1, 1] due to their equioscillation behavior.
    They minimize the maximum deviation, providing better conditioning than
    Legendre polynomials for capturing features like sin(x) or exponential terms.
    """
    # Calculate Chebyshev polynomials for x and y
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2*t^2 - 1
    # T_3(t) = 4*t^3 - 3*t
    
    Px0 = np.ones_like(x)
    Px1 = x
    Px2 = 2 * x**2 - 1
    Px3 = 4 * x**3 - 3 * x
    
    Py0 = np.ones_like(y)
    Py1 = y
    Py2 = 2 * y**2 - 1
    Py3 = 4 * y**3 - 3 * y
    
    # Construct the 10 basis terms
    # Order: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), (3,0), (2,1), (1,2), (0,3)
    
    b0 = Px0 * Py0
    b1 = Px1 * Py0
    b2 = Px0 * Py1
    b3 = Px2 * Py0
    b4 = Px1 * Py1
    b5 = Px0 * Py2
    b6 = Px3 * Py0
    b7 = Px2 * Py1
    b8 = Px1 * Py2
    b9 = Px0 * Py3
    
    # Combine with parameters
    # params is an array of length 10
    # We assume params[0] corresponds to b0, etc.
    
    result = (params[0] * b0 + 
              params[1] * b1 + 
              params[2] * b2 + 
              params[3] * b3 + 
              params[4] * b4 + 
              params[5] * b5 + 
              params[6] * b6 + 
              params[7] * b7 + 
              params[8] * b8 + 
              params[9] * b9)
              
    return result
# EVOLVE-BLOCK-END
