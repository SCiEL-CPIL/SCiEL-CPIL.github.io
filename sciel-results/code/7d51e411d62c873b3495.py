import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function model with quadratic numerator and denominator.
    params: p0..p5 for numerator, q1..q4 for denominator (denom = 1 + ...)
    """
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, q1, q2, q3, q4 = params
    
    # Precompute basis terms
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    # Numerator: quadratic polynomial
    numerator = p0 + p1 * x + p2 * y + p3 * x2 + p4 * y2 + p5 * xy
    
    # Denominator: 1 + quadratic polynomial (constant term fixed at 1 for identifiability)
    denominator = 1.0 + q1 * x + q2 * y + q3 * x2 + q4 * y2
    
    # Safety: prevent division by zero or near-zero
    # Use a smooth safeguard to avoid non-differentiability issues in BFGS
    # If denominator is very small, clip it smoothly
    eps = 1e-6
    
    # Smooth clip: denom_safe = denominator * (1 - exp(-abs(denominator)/eps)) + sign(denominator)*eps*exp(-abs(denominator)/eps)
    # Simpler approach: just add a tiny regularization to keep it away from 0
    # But that biases the model. Let's use a where clause with a small epsilon.
    # To be differentiable, we can use: denom_safe = denominator + eps * (denominator / (abs(denominator) + eps))
    # This is approximately denominator when |denominator| >> eps, and ~eps when |denominator| ~ 0
    
    abs_denom = np.abs(denominator)
    # Smooth regularization term
    reg_term = eps * (denominator / (abs_denom + eps))
    denom_safe = denominator + reg_term
    
    # Ensure no NaN/Inf
    pred = numerator / denom_safe
    
    # Replace any non-finite values with 0 (shouldn't happen with the safeguard)
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
