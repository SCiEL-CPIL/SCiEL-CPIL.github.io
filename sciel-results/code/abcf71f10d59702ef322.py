"""Symbolic regression with a rational function model using Chebyshev polynomials."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a rational function model:
    y(x) = P(x) / Q(x)
    
    where P(x) and Q(x) are linear combinations of Chebyshev polynomials of degree 0-4.
    
    10 parameters correspond to:
      p[0]..p[4]: coefficients of numerator P(x) = sum_{k=0}^{4} p[k] * T_k(x/2)
      p[5]..p[9]: coefficients of denominator Q(x) = sum_{k=0}^{4} p[5+k] * T_k(x/2)
    
    This rational form can represent functions with poles, asymptotes, and
    rapid local changes. Using Chebyshev polynomials on the scaled interval [-1,1]
    improves numerical conditioning compared to monomials.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
    
    # Scale x to [-1, 1]
    x_scaled = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_4
    # T_0(x) = 1
    # T_1(x) = x
    # T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
    
    T0 = np.ones_like(x_scaled)
    T1 = x_scaled
    T2 = 2.0 * x_scaled * T1 - T0
    T3 = 2.0 * x_scaled * T2 - T1
    T4 = 2.0 * x_scaled * T3 - T2
    
    # Numerator polynomial P(x)
    numerator = p[0] * T0 + p[1] * T1 + p[2] * T2 + p[3] * T3 + p[4] * T4
    
    # Denominator polynomial Q(x)
    denominator = p[5] * T0 + p[6] * T1 + p[7] * T2 + p[8] * T3 + p[9] * T4
    
    # Compute result with safe division
    # Add small epsilon to denominator to avoid division by zero
    # Use sign-aware epsilon to preserve the sign of the denominator
    eps = 1e-10
    safe_denom = denominator + np.sign(denominator) * eps
    # Handle case where denominator is exactly zero
    safe_denom = np.where(denominator == 0, eps, safe_denom)
    
    result = numerator / safe_denom
    
    # Replace any non-finite values with a large finite value
    result = np.where(np.isfinite(result), result, 1e6 * np.sign(numerator))
    # Handle case where numerator is zero and result became non-finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
