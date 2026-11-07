"""Nonlinear oscillator acceleration via Gaussian Process-inspired kernel basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a sum of smooth Gaussian bumps
    in the (t, x, v) state space.
    
    The 10 parameters define:
    - 3 Gaussian centers (in t, x, v dimensions)
    - 3 Gaussian scales (inverse bandwidths)
    - 3 Gaussian weights
    - 1 global offset
    
    This kernel-based approach can capture arbitrary nonlinear acceleration
    fields without assuming specific polynomial or rational structures.
    """
    # Extract parameters
    ct = params[0]  # center in t
    cx = params[1]  # center in x
    cv = params[2]  # center in v
    st = params[3]  # scale in t
    sx = params[4]  # scale in x
    sv = params[5]  # scale in v
    wt = params[6]  # weight for t-dependent term
    wx = params[7]  # weight for x-dependent term
    wv = params[8]  # weight for v-dependent term
    offset = params[9]  # global offset
    
    # Ensure scales are positive by using abs
    st_abs = np.abs(st) + 1e-6
    sx_abs = np.abs(sx) + 1e-6
    sv_abs = np.abs(sv) + 1e-6
    
    # Compute Gaussian kernels
    # Time-dependent component
    dt = (t - ct) / st_abs
    gauss_t = np.exp(-0.5 * dt * dt)
    
    # Position-dependent component  
    dx = (x - cx) / sx_abs
    gauss_x = np.exp(-0.5 * dx * dx)
    
    # Velocity-dependent component
    dv = (v - cv) / sv_abs
    gauss_v = np.exp(-0.5 * dv * dv)
    
    # Combine: sum of weighted Gaussian contributions
    # Each term captures acceleration associated with different state aspects
    result = offset + wt * gauss_t + wx * gauss_x + wv * gauss_v
    
    return result
# EVOLVE-BLOCK-END
