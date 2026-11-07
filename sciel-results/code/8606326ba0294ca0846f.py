import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression: recover an unknown function of 1 input.
    
    Architecture: Orthonormal Legendre polynomial basis (degree 0-3) +
    learnable-frequency sine + learnable-frequency cosine + exponential term.
    
    Parameters (10 total):
      params[0]: constant (P_0 coefficient)
      params[1]: linear (P_1 coefficient)
      params[2]: quadratic (P_2 coefficient)
      params[3]: cubic (P_3 coefficient)
      params[4]: amplitude of sin(freq_sin * x)
      params[5]: frequency of sin(freq_sin * x)
      params[6]: amplitude of cos(freq_cos * x)
      params[7]: frequency of cos(freq_cos * x)
      params[8]: rate of exp(rate_exp * x)
      params[9]: amplitude of exp(rate_exp * x)
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Extract parameters
    c0 = p[0]
    c1 = p[1]
    c2 = p[2]
    c3 = p[3]
    amp_sin = p[4]
    freq_sin = p[5]
    amp_cos = p[6]
    freq_cos = p[7]
    rate_exp = p[8]
    amp_exp = p[9]
    
    # Map x from [-3, 3] to t in [-1, 1] for Legendre polynomials
    t = x / 3.0
    
    # Legendre Polynomials P_n(t)
    P0 = np.ones_like(t)
    P1 = t
    P2 = 0.5 * (3.0 * t**2 - 1.0)
    P3 = 0.5 * (5.0 * t**3 - 3.0 * t)
    
    # Polynomial component
    y_poly = c0 * P0 + c1 * P1 + c2 * P2 + c3 * P3
    
    # Clip frequencies for numerical stability
    freq_sin = np.clip(freq_sin, -20.0, 20.0)
    freq_cos = np.clip(freq_cos, -20.0, 20.0)
    
    # Trigonometric terms
    y_sin = amp_sin * np.sin(freq_sin * x)
    y_cos = amp_cos * np.cos(freq_cos * x)
    
    # Exponential term with clipping
    rate_exp = np.clip(rate_exp, -10.0, 10.0)
    exp_arg = rate_exp * x
    exp_arg = np.clip(exp_arg, -30.0, 30.0)
    y_exp = amp_exp * np.exp(exp_arg)
    
    # Combine
    result = y_poly + y_sin + y_cos + y_exp
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
