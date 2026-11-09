"""Symbolic regression with Chebyshev + RBF mixed basis for smooth and local features."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: linear combination of 10 basis functions of x.
    
    Combines:
    - Chebyshev polynomials (well-conditioned global polynomial approximation)
    - Gaussian RBFs (localized features for kinks/non-smooth behavior)
    - tanh (saturating nonlinearity)
    
    Basis:
      0: T_0(t) = 1
      1: T_1(t) = t
      2: T_2(t) = 2t^2 - 1
      3: T_3(t) = 4t^3 - 3t
      4: T_4(t) = 8t^4 - 8t^2 + 1
      5: T_5(t) = 16t^5 - 20t^3 + 5t
      6: exp(-(x+1.5)^2 / 2s^2)  RBF centered at -1.5
      7: exp(-(x-1.5)^2 / 2s^2)  RBF centered at  1.5
      8: tanh(x)
      9: sin(x)
    where t = x/3 maps [-3,3] to [-1,1]
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map to [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Chebyshev polynomials via recurrence
    # T_0(t) = 1
    # T_1(t) = t
    # T_{k+1}(t) = 2t * T_k(t) - T_{k-1}(t)
    
    T0 = np.ones_like(t)
    T1 = t.copy()
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    T5 = 2.0 * t * T4 - T3
    
    # Gaussian RBFs with moderate width for localized features
    # sigma = 0.8, so sigma^2 = 0.64, inv_2s2 = 1/(2*0.64)
    inv_2s2 = 1.0 / (2.0 * 0.64)
    
    # RBF centered at -1.5
    rbf_neg = np.exp(-inv_2s2 * (x - (-1.5)) ** 2)
    
    # RBF centered at 1.5
    rbf_pos = np.exp(-inv_2s2 * (x - 1.5) ** 2)
    
    # Global smooth terms
    tanh_term = np.tanh(x)
    sin_term = np.sin(x)
    
    # Linear combination of all 10 basis functions
    y = (params[0] * T0 +
         params[1] * T1 +
         params[2] * T2 +
         params[3] * T3 +
         params[4] * T4 +
         params[5] * T5 +
         params[6] * rbf_neg +
         params[7] * rbf_pos +
         params[8] * tanh_term +
         params[9] * sin_term)
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
        
    return y
# EVOLVE-BLOCK-END
