import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a hybrid basis: Chebyshev polynomials,
    multi-frequency trigonometric functions, and a tanh saturation term.
    
    Strategy:
    - Use Chebyshev polynomials T_0..T_6 on normalized x in [-1, 1] for
      well-conditioned polynomial approximation.
    - Use Sin/Cos with frequencies 1, 2, 3 to capture periodic/oscillatory
      behavior.
    - Use a Tanh term to capture saturation or step-like transitions.
    - This provides a rich, diverse basis that can approximate a wide range
      of smooth functions, with BFGS fitting the 10 linear weights.
    """
    x_flat = x.flatten()
    
    # Normalize x to [-1, 1] for Chebyshev polynomials
    x_norm = x_flat / 2.0
    
    # Build basis functions (10 total)
    basis = []
    
    # Chebyshev polynomials T_0 through T_6 (7 terms)
    # T_0 = 1
    basis.append(np.ones_like(x_norm))
    # T_1 = x
    basis.append(x_norm.copy())
    # T_n = 2*x*T_{n-1} - T_{n-2}
    T_prev2 = basis[0]  # T_0
    T_prev1 = basis[1]  # T_1
    for n in range(2, 7):
        T_curr = 2.0 * x_norm * T_prev1 - T_prev2
        basis.append(T_curr)
        T_prev2 = T_prev1
        T_prev1 = T_curr
    
    # Trig functions: sin(x), cos(x), sin(2x), cos(2x) (4 terms)
    basis.append(np.sin(x_flat))
    basis.append(np.cos(x_flat))
    basis.append(np.sin(2.0 * x_flat))
    basis.append(np.cos(2.0 * x_flat))
    
    # Tanh saturation term (1 term)
    basis.append(np.tanh(x_flat))
    
    # Stack into matrix of shape (N, 10)
    B = np.column_stack(basis)
    
    # Linear combination with params
    predictions = B @ params
    
    return predictions.reshape(x.shape)
