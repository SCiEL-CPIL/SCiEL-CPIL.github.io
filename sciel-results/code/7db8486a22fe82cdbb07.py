"""Gaussian Process-inspired kernel basis for nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled as a Gaussian bump with independent scales
    plus linear terms for position and velocity.
    
    Parameters:
        t: time (input 1)
        x: position (input 2)
        v: velocity (input 3)
        params: array of 10 parameters.
                params[0]   = center t (normalized)
                params[1]   = center x (normalized)
                params[2]   = center v (normalized)
                params[3]   = scale t (sigma_t)
                params[4]   = scale x (sigma_x)
                params[5]   = scale v (sigma_v)
                params[6]   = weight (w)
                params[7]   = linear coeff x (c_x)
                params[8]   = linear coeff v (c_v)
                params[9]   = bias (b)
    
    Returns:
        Acceleration predictions (same shape as t)
    """
    # Fixed normalization constants based on training ranges
    # t in [30, 50] -> mean 40, scale 10
    # x in [-0.2056, 0.1749] -> mean ~0, scale 0.2
    # v in [-0.3391, 0.3624] -> mean ~0, scale 0.35
    t_norm = (t - 40.0) / 10.0
    x_norm = x / 0.2
    v_norm = v / 0.35
    
    # Extract parameters
    t_c = params[0]
    x_c = params[1]
    v_c = params[2]
    
    sigma_t = np.maximum(np.abs(params[3]), 1e-4)
    sigma_x = np.maximum(np.abs(params[4]), 1e-4)
    sigma_v = np.maximum(np.abs(params[5]), 1e-4)
    
    w = params[6]
    c_x = params[7]
    c_v = params[8]
    bias = params[9]
    
    # Compute Gaussian term with independent scales
    # g = w * exp(-0.5 * ((t_n - t_c)/sigma_t)^2 - 0.5 * ((x_n - x_c)/sigma_x)^2 - 0.5 * ((v_n - v_c)/sigma_v)^2)
    
    dt = t_norm - t_c
    dx = x_norm - x_c
    dv = v_norm - v_c
    
    exponent = -0.5 * ((dt / sigma_t)**2 + (dx / sigma_x)**2 + (dv / sigma_v)**2)
    
    # Clip exponent to avoid underflow issues (though exp handles it, clipping is safe)
    exponent = np.clip(exponent, -500, 0)
    
    gauss_term = w * np.exp(exponent)
    
    # Linear terms using raw x and v to preserve physical scale for coefficients
    linear_term = c_x * x + c_v * v
    
    return bias + gauss_term + linear_term
# EVOLVE-BLOCK-END
