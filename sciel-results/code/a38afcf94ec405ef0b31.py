# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluates the model using a rich basis of Chebyshev polynomials,
    trigonometric, hyperbolic, and Gaussian functions.
    params[0..9] are coefficients for:
    1, T1(x/3), T2(x/3), T3(x/3), T4(x/3), T5(x/3), sin(x), cos(x), tanh(x), exp(-x^2)
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute scaled variable for Chebyshev polynomials on [-1, 1]
    t = x / 3.0
    
    # Chebyshev polynomials of the first kind
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2*t^2 - 1
    # T3(t) = 4*t^3 - 3*t
    # T4(t) = 8*t^4 - 8*t^2 + 1
    # T5(t) = 16*t^5 - 20*t^3 + 5*t
    
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    t5 = t4 * t
    
    T1 = t
    T2 = 2.0 * t2 - 1.0
    T3 = 4.0 * t3 - 3.0 * t
    T4 = 8.0 * t4 - 8.0 * t2 + 1.0
    T5 = 16.0 * t5 - 20.0 * t3 + 5.0 * t
    
    # Other basis functions
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    tanh_x = np.tanh(x)
    exp_neg_x2 = np.exp(-x * x)
    
    # Combine
    y = (p0 * 1.0 +
         p1 * T1 +
         p2 * T2 +
         p3 * T3 +
         p4 * T4 +
         p5 * T5 +
         p6 * sin_x +
         p7 * cos_x +
         p8 * tanh_x +
         p9 * exp_neg_x2)
         
    return y
# EVOLVE-BLOCK-END
