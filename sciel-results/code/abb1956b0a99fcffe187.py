"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# The evaluator calls equation(t, x, v, params).
# Hybrid model: Physics-informed oscillator structure with normalized variables
# and a richer basis that includes time-state cross terms.
#
# Structure (10 params):
#   Normalized variables:
#     xn = (x - cx) / hx,  vn = (v - cv) / hv
#   Time terms:
#     s = sin(t), c = cos(t)
#
#   y = p0*s + p1*c                          # Pure forcing
#       + p2*xn                              # Linear restoring
#       + p3*xn^3                            # Duffing cubic
#       + p4*vn                              # Linear damping
#       + p5*xn^2*vn                         # Van der Pol
#       + p6*vn^3                            # Cubic damping
#       + p7*s*xn                            # Time-state cross
#       + p8*c*xn                            # Time-state cross
#       + p9*xn*vn                           # Quadratic coupling
#
# Normalization improves conditioning for BFGS.
# Cross terms p7*s*xn and p8*c*xn allow the forcing to modulate with displacement,
# which is more general than pure additive forcing.

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid physics-informed nonlinear oscillator model.
    
    Uses normalized state variables and a basis that combines:
      - Periodic forcing (sin, cos)
      - Polynomial restoring force (linear, cubic)
      - State-dependent damping (linear, Van der Pol, cubic)
      - Time-state cross terms
      - Quadratic coupling
    """
    # Normalization constants from training ranges
    # x: [-0.2056, 0.1749] -> center = -0.01535, half_range = 0.19025
    # v: [-0.3391, 0.3624] -> center = 0.01165, half_range = 0.35075
    cx = -0.01535
    hx = 0.19025
    cv = 0.01165
    hv = 0.35075
    
    # Normalize
    xn = (x - cx) / hx
    vn = (v - cv) / hv
    
    # Precompute powers
    xn2 = xn * xn
    xn3 = xn2 * xn
    vn2 = vn * vn
    vn3 = vn2 * vn
    xn_vn = xn * vn
    xn2_vn = xn2 * vn
    
    # Time terms
    s = np.sin(t)
    c = np.cos(t)
    
    # Build prediction
    y = (params[0] * s
         + params[1] * c
         + params[2] * xn
         + params[3] * xn3
         + params[4] * vn
         + params[5] * xn2_vn
         + params[6] * vn3
         + params[7] * s * xn
         + params[8] * c * xn
         + params[9] * xn_vn)
    
    return y
# EVOLVE-BLOCK-END
