"""Symbolic regression for a nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Mixture Model-inspired kernel basis for nonlinear oscillator acceleration.
    
    This model assumes the acceleration field in the (x, v) phase space can be
    approximated by a sum of smooth Gaussian bumps, plus a periodic time-dependent
    forcing term. This structure is more flexible than a fixed polynomial (like Duffing)
    for capturing localized nonlinear features, but uses parameter transformations
    (tanh for centers, exp for widths) to ensure BFGS convergence from an initial
    guess of all ones.
    
    Parameters:
    p0: weight of bump 1
    p1: center x of bump 1 (scaled via tanh)
    p2: center v of bump 1 (scaled via tanh)
    p3: log-width of bump 1 (scaled via exp)
    p4: weight of bump 2
    p5: center x of bump 2 (scaled via tanh)
    p6: center v of bump 2 (scaled via tanh)
    p7: log-width of bump 2 (scaled via exp)
    p8: amplitude of periodic time term sin(t_scaled)
    p9: bias term
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale inputs to roughly [-1, 1] for better conditioning
    # t in [30, 50] -> center 40, range 10
    # x in [-0.2056, 0.1749] -> approx range 0.4, center ~0
    # v in [-0.3391, 0.3624] -> approx range 0.7, center ~0
    
    ts = (t - 40.0) / 10.0
    xs = x / 0.2
    vs = v / 0.35
    
    # Transform parameters to valid ranges
    # Centers: map to [-2, 2] to cover the scaled input range with margin
    cx1 = 2.0 * np.tanh(p1)
    cv1 = 2.0 * np.tanh(p2)
    cx2 = 2.0 * np.tanh(p5)
    cv2 = 2.0 * np.tanh(p6)
    
    # Widths: ensure positive and not too small to avoid division by zero or extreme gradients
    # exp(p) where p starts at 1 -> width ~ 2.7, which is good (wider than data range)
    # Clamp to prevent underflow/overflow issues in the exponent calculation
    sigma1 = np.exp(np.clip(p3, -10, 10))
    sigma2 = np.exp(np.clip(p7, -10, 10))
    
    # Calculate Gaussian bumps
    # G = exp(-0.5 * (dx^2 + dv^2) / sigma^2)
    dx1 = xs - cx1
    dv1 = vs - cv1
    dist1_sq = dx1**2 + dv1**2
    G1 = np.exp(-0.5 * dist1_sq / (sigma1**2))
    
    dx2 = xs - cx2
    dv2 = vs - cv2
    dist2_sq = dx2**2 + dv2**2
    G2 = np.exp(-0.5 * dist2_sq / (sigma2**2))
    
    # Time-dependent term
    # Using sin(ts) as a generic periodic forcing. 
    # If the oscillator is driven, this captures the phase.
    time_term = p8 * np.sin(ts)
    
    # Combine
    a = p9 + p0 * G1 + p4 * G2 + time_term
    
    # Ensure finite output
    if not np.all(np.isfinite(a)):
        return np.zeros_like(a)
        
    return a
# EVOLVE-BLOCK-END
