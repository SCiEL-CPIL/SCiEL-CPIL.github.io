"""Symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Soft-thresholded Chebyshev polynomial basis with 10 parameters.
    
    Each Chebyshev term T_k(t) is multiplied by a learnable soft-clipping factor
    (1 + tanh(p_i * (|t| - c_i))) where p_i and c_i are derived from the params.
    This creates piecewise-linear-like flexibility to handle potential non-smooth
    features (kinks or saturation) that global smooth bases struggle with.
    
    Parameter layout (10 params):
    - params[0:5]: coefficients for T_0 through T_4
    - params[5:10]: coefficients for T_5 through T_9
    
    The soft-clipping uses fixed centers c_k = k/9 (spaced across [-1,1] range of |t|)
    and sharpness p_k = 5.0 + k (increasing sharpness for higher-order terms).
    This design keeps the basis smooth (tanh is C^infty) while introducing
    localized sensitivity changes that can better fit functions with kinks.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1] for stability
    t = x / 3.0
    t_safe = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Compute Chebyshev polynomials T_0 to T_9 using recurrence
    T = np.empty((10, len(t_safe)), dtype=np.float64)
    T[0] = np.ones_like(t_safe)
    T[1] = t_safe.copy()
    for n in range(2, 10):
        T[n] = 2.0 * t_safe * T[n-1] - T[n-2]
    
    # Apply soft-thresholding: multiply each T_k by (1 + tanh(p_k * (|t| - c_k)))
    # c_k = k/9 spreads centers across [0, 1] (since |t| in [0,1])
    # p_k = 5.0 + k gives increasing sharpness for higher-order terms
    # This makes higher-order terms more sensitive to the magnitude of t,
    # creating a piecewise-linear-like response that can capture kinks
    
    abs_t = np.abs(t_safe)
    result = np.zeros_like(t_safe)
    for k in range(10):
        c_k = k / 9.0  # center in [0, 1]
        p_k = 5.0 + k  # sharpness, increasing with order
        soft_factor = 1.0 + np.tanh(p_k * (abs_t - c_k))
        result += params[k] * T[k] * soft_factor
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
