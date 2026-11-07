"""
Non-Linear Parametric Family for Symbolic Regression.

Strategy:
- Instead of a linear basis (where BFGS only adjusts weights), we define a 
  non-linear functional form where the parameters define the shape of the curve.
- We use a Sum of Exponentials: f(x) = sum_{i=1}^5 A_i * exp(-(x - c_i)^2 / w_i).
  This form is flexible enough to approximate Gaussians, peaks, and even 
  polynomial-like shapes (by using wide widths), but is particularly strong 
  for localized features that polynomials struggle with.
- The 10 parameters are: [A1, c1, w1, A2, c2, w2, A3, c3, w3, A4, c4, w4, A5, c5, w5]
  Wait, 5 terms * 3 params = 15 params. We only have 10.
  
  Revised Strategy: Use 3 terms of exponentials (9 params) + 1 global offset? 
  Or 2 terms of a more complex basis?
  
  Let's use a "Gaussian Mixture" with 3 components.
  Params: [A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3, offset]
  
  Why this works:
  1. If the target is a single Gaussian, 1 component fits it perfectly.
  2. If the target is a polynomial, wide sigmas can approximate it, but it's 
     less efficient than Chebyshev. However, if the target is NON-polynomial 
     (e.g., exp(-x^2), tanh(x), 1/(1+x^2)), this family can fit it much better 
     than any fixed polynomial basis of degree 4.
  3. BFGS can navigate the non-convex landscape of centers/widths to find 
     the correct structure.
  
  To improve robustness, we ensure sigma is always positive and not too small 
  to avoid numerical instability (e.g., sigma = 0.1 + |raw_sigma|).
  
  Alternative: Rational function P(x)/Q(x) with degree 2/2.
  Params: [a0, a1, a2, b1, b2, offset] -> 6 params? No, we need 10.
  Degree 3/2? 
  Let's stick to the Gaussian Mixture as it's smooth and stable.
  
  Actually, let's try a hybrid: 
  1. A global polynomial trend (degree 2) using 3 params.
  2. Two Gaussian bumps using 4 params (A, mu, sigma for each).
  3. One more param for a linear slope or offset?
  
  Let's use 2 Gaussians + a Quadratic + a Linear + Offset?
  f(x) = a*x^2 + b*x + c + A1*exp(-(x-mu1)^2/s1^2) + A2*exp(-(x-mu2)^2/s2^2)
  Params: [a, b, c, A1, mu1, s1, A2, mu2, s2, offset] -> 10 params.
  
  This combines the global trend capability of polynomials with the local 
  flexibility of Gaussians, but crucially, the Gaussian centers and widths are 
  OPTIMIZED, not fixed. This is the key structural change from the previous 
  program where centers were fixed.
"""

import numpy as np
import time as _time

_START = _time.time()
_DEADLINE = _START + 140.0  # Safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points
    params: array of 10 coefficients
    Returns: predicted values
    """
    # Deadline guard
    if _time.time() > _DEADLINE:
        # Return a safe fallback: mean if we're out of time
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            x = x.ravel()
        p = np.asarray(params, dtype=float).ravel()
        if p.size >= 1:
            return np.full(len(x), p[0])
        return np.zeros(len(x))
    
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        x = x.ravel()
        
    p = np.asarray(params, dtype=float).ravel()
    if p.size != 10:
        # Fallback if params shape is wrong
        p = np.ones(10)
        
    N = len(x)
    if N == 0:
        return np.array([])
        
    # Unpack parameters
    # f(x) = a*x^2 + b*x + c + A1*exp(-(x-mu1)^2/(2*s1^2)) + A2*exp(-(x-mu2)^2/(2*s2^2)) + offset
    # Indices:
    # 0: a (quad coeff)
    # 1: b (linear coeff)
    # 2: c (constant)
    # 3: A1 (amp 1)
    # 4: mu1 (center 1)
    # 5: s1 (width 1)
    # 6: A2 (amp 2)
    # 7: mu2 (center 2)
    # 8: s2 (width 2)
    # 9: offset
    
    a, b, c, A1, mu1, s1, A2, mu2, s2, offset = p
    
    # Ensure widths are positive and bounded to prevent overflow/underflow
    # s1, s2 are raw params. We use softplus-like mapping or abs+epsilon.
    # Using abs ensures differentiability almost everywhere, but abs(0) is 0.
    # Better: s = 0.1 + np.abs(s_raw) to ensure minimum width.
    # However, BFGS uses gradients. abs has a discontinuous gradient at 0.
    # Let's use s = 0.1 + np.exp(s_raw) ? No, that might grow too fast.
    # Let's use s = 0.1 + np.abs(s_raw). It's robust enough for BFGS.
    
    s1_val = 0.1 + np.abs(s1)
    s2_val = 0.1 + np.abs(s2)
    
    # Compute polynomial part
    poly_part = a * x**2 + b * x + c
    
    # Compute Gaussian parts
    # Clip the exponent argument to prevent overflow in exp
    # exp(-large) -> 0, exp(large) -> inf. We want exp(- (x-mu)^2 / (2*s^2))
    # The argument is always <= 0. So no overflow to +inf. Underflow to 0 is fine.
    
    diff1 = x - mu1
    gauss1 = A1 * np.exp(-(diff1**2) / (2.0 * s1_val**2))
    
    diff2 = x - mu2
    gauss2 = A2 * np.exp(-(diff2**2) / (2.0 * s2_val**2))
    
    result = poly_part + gauss1 + gauss2 + offset
    
    # Ensure finite values
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
