"""
Strategy: Hybrid Global Basis (Chebyshev + Fourier).

This strategy uses a hybrid basis combining Chebyshev polynomials and Fourier terms.
The Rationale:
1. Flexibility: The target function might have both smooth non-periodic components 
   (well-captured by Chebyshev) and periodic components (well-captured by Fourier).
2. Better Conditioning: Chebyshev polynomials are better conditioned than monomials.
   Fourier terms are orthogonal on the interval.
3. Linear in Parameters: The model remains linear in the weights, allowing BFGS 
   (or even closed-form least squares) to find the global optimum efficiently.
4. 10 Parameters: We allocate 5 parameters to Chebyshev (degrees 0-4) and 5 to 
   Fourier (sine/cosine of fundamental and harmonics). This balances global and 
   periodic fitting capabilities.

Basis functions:
- T_0(t) = 1
- T_1(t) = t
- T_2(t) = 2t^2 - 1
- T_3(t) = 4t^3 - 3t
- T_4(t) = 8t^4 - 8t^2 + 1
- sin(pi * x / 3)
- cos(pi * x / 3)
- sin(2*pi * x / 3)
- cos(2*pi * x / 3)
- sin(3*pi * x / 3)  (or cos, but let's mix)

Actually, let's use:
- 5 Chebyshev terms (T_0 to T_4)
- 5 Fourier terms: sin(pi x/3), cos(pi x/3), sin(2 pi x/3), cos(2 pi x/3), sin(3 pi x/3)

This covers a wide range of smooth functions.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the hybrid Chebyshev + Fourier expansion.
    
    Args:
        x: Input array of shape (N,)
        params: Coefficients array of shape (10,)
    
    Returns:
        Prediction array of shape (N,)
    """
    x = np.asarray(x).ravel()
    N = len(x)
    
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0
    
    # Initialize basis matrix
    basis = np.zeros((N, 10))
    
    # --- Chebyshev Polynomials (T_0 to T_4) ---
    # T_0(t) = 1
    basis[:, 0] = 1.0
    
    # T_1(t) = t
    basis[:, 1] = t
    
    # T_2(t) = 2t^2 - 1
    basis[:, 2] = 2.0 * t**2 - 1.0
    
    # T_3(t) = 4t^3 - 3t
    basis[:, 3] = 4.0 * t**3 - 3.0 * t
    
    # T_4(t) = 8t^4 - 8t^2 + 1
    basis[:, 4] = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    # --- Fourier Terms ---
    # Let theta = pi * x / 3. As x ranges [-3, 3], theta ranges [-pi, pi].
    theta = np.pi * x / 3.0
    
    # 5th param: sin(theta)
    basis[:, 5] = np.sin(theta)
    
    # 6th param: cos(theta)
    basis[:, 6] = np.cos(theta)
    
    # 7th param: sin(2*theta)
    basis[:, 7] = np.sin(2.0 * theta)
    
    # 8th param: cos(2*theta)
    basis[:, 8] = np.cos(2.0 * theta)
    
    # 9th param: sin(3*theta)
    basis[:, 9] = np.sin(3.0 * theta)
    
    # Linear combination: y = basis @ params
    return basis @ params

# EVOLVE-BLOCK-END
