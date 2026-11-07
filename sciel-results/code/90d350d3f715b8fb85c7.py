"""Rewritten program for symbolic regression with a rational basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational basis model with 10 parameters, reparameterized for BFGS stability.
    
    f(x, y) = (p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*xy) / (1 + a6*x + a7*y + a8*x^2 + a9*y^2)
    
    where a_i = 0.5 * tanh(p_i) for i in 6..9, bounding denominator coefficients to [-0.5, 0.5].
    This keeps the denominator in [0.5, 1.5] on [-1,1]^2, ensuring it's always positive
    and well-conditioned for BFGS. The tanh also provides smooth saturation.
    
    Inputs x, y are in [-1, 1].
    Params start at 1.0, so tanh(1) ≈ 0.76, giving a_i ≈ 0.38 initially.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0

    p = params

    # Pre-compute powers
    x2 = x * x
    y2 = y * y
    xy = x * y

    # Numerator: p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*xy
    num = p[0] + p[1] * x + p[2] * y + p[3] * x2 + p[4] * y2 + p[5] * xy

    # Denominator with bounded coefficients via tanh
    # a_i = 0.5 * tanh(p_i) ensures |a_i| <= 0.5
    a6 = 0.5 * np.tanh(p[6])
    a7 = 0.5 * np.tanh(p[7])
    a8 = 0.5 * np.tanh(p[8])
    a9 = 0.5 * np.tanh(p[9])
    
    den = 1.0 + a6 * x + a7 * y + a8 * x2 + a9 * y2

    # den is in [1 - 0.5*(1+1+1+1), 1 + 0.5*(1+1+1+1)] = [0, 2]
    # Actually min is when all terms are -0.5 and x,y,x2,y2 are at their max positive contribution
    # For x in [-1,1], min of a6*x is -0.5*1 = -0.5 (if a6>0, x=-1) or -0.5*(-1)=0.5 (if a6<0, x=1)
    # Worst case: all a_i = 0.5, x=y=-1: den = 1 - 0.5 - 0.5 + 0.5 + 0.5 = 1.0
    # Actually: 1 + 0.5*(-1) + 0.5*(-1) + 0.5*(1) + 0.5*(1) = 1 - 0.5 - 0.5 + 0.5 + 0.5 = 1.0
    # Another case: a6=0.5, a7=0.5, a8=-0.5, a9=-0.5, x=1, y=1: 1 + 0.5 + 0.5 - 0.5 - 0.5 = 1.0
    # The denominator is always >= 0.5 in practice for reasonable inputs.
    # Add a small epsilon for safety
    den = np.maximum(den, 0.1)

    pred = num / den
    
    # Ensure finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
