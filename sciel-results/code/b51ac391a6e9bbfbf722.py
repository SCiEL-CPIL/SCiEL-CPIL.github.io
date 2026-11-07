"""
Strategy: Global Hybrid Fourier-Polynomial Basis (Linear in Parameters)

Rationale:
1. The previous RBF approach suffered from non-convexity (optimizing centers/widths).
2. Inspirations show that linear-in-parameters models (Chebyshev/Polynomial) score ~7.0, 
   significantly higher than the RBF score of 4.5.
3. Knowledge suggests Fourier bases are better for oscillatory targets.
4. Therefore, we use a fixed basis that combines low-degree polynomials and low-frequency 
   Fourier terms. Since the model is linear in parameters (weights), BFGS minimizes the 
   convex quadratic loss function to the global optimum immediately.

Basis Functions (10 terms):
1. 1.0 (Bias)
2. x
3. y
4. x * y
5. x**2
6. y**2
7. sin(2*pi*x)
8. cos(2*pi*x)
9. sin(2*pi*y)
10. cos(2*pi*y)

This basis is orthonormal-ish over [-1, 1] for the polynomial parts and captures 
fundamental frequencies for the Fourier parts. It is robust to the specific 
distribution of the hidden function, provided it is not highly complex or 
chaotic.
"""

import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 fixed basis functions.
    params: array of length 10, [w0, w1, ..., w9]
    """
    if params is None or len(params) < 10:
        return np.zeros_like(x)
    
    # Extract weights
    w = params[:10]
    
    # Compute basis functions
    # Polynomial terms
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * y
    b4 = x * x
    b5 = y * y
    
    # Fourier terms (fundamental frequency 2*pi maps [-1,1] to [0, 2pi])
    # Note: Using 2*pi * x ensures one full period across the domain [-1, 1]
    # If the function has higher frequencies, this might miss them, but 
    # polynomials help smooth trends.
    b6 = np.sin(2.0 * np.pi * x)
    b7 = np.cos(2.0 * np.pi * x)
    b8 = np.sin(2.0 * np.pi * y)
    b9 = np.cos(2.0 * np.pi * y)
    
    # Combine linearly
    # result = w0*b0 + w1*b1 + ... + w9*b9
    result = (w[0] * b0 + 
              w[1] * b1 + 
              w[2] * b2 + 
              w[3] * b3 + 
              w[4] * b4 + 
              w[5] * b5 + 
              w[6] * b6 + 
              w[7] * b7 + 
              w[8] * b8 + 
              w[9] * b9)
              
    return result
