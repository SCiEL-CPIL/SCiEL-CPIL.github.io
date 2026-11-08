"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator, from position and velocity.
    Uses a physically-motivated basis with well-scaled features for BFGS convergence.
    """
    p = params
    
    # Scale inputs to be O(1) in the training range
    # x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    x_s = x * 1.5   # ~[-0.86, 0.79]
    v_s = v * 4.0   # ~[-0.70, 1.13]
    
    # Pre-compute basis functions
    x2 = x_s * x_s
    x3 = x2 * x_s
    x5 = x3 * x2
    v2 = v_s * v_s
    v3 = v2 * v_s
    v5 = v3 * v2
    
    # Cross terms
    xv = x_s * v_s
    x2v = x2 * v_s
    x3v = x3 * v_s
    
    # Smooth nonlinear saturating functions
    tanh_x = np.tanh(x_s)
    tanh_v = np.tanh(v_s)
    tanh_x3 = np.tanh(x3)
    tanh_v3 = np.tanh(v3)
    
    # Sinusoidal terms for periodic nonlinearity
    sin_x = np.sin(x_s)
    sin_x3 = np.sin(x3)
    
    # Linear combination of 10 features
    # Restoring force terms (odd in x):
    # 0: x_s (linear restoring)
    # 1: x3 (cubic restoring, Duffing-like)
    # 2: tanh_x (softening stiffness)
    # Damping terms (odd in v):
    # 3: v_s (linear damping)
    # 4: v3 (cubic damping)
    # 5: tanh_v (softening damping)
    # Cross/coupling terms:
    # 6: xv (parametric coupling)
    # 7: x2v (velocity-dependent stiffness)
    # 8: sin_x (periodic stiffness)
    # 9: 1 (bias)
    
    y = (p[0] * x_s +
         p[1] * x3 +
         p[2] * tanh_x +
         p[3] * v_s +
         p[4] * v3 +
         p[5] * tanh_v +
         p[6] * xv +
         p[7] * x2v +
         p[8] * sin_x +
         p[9])
         
    return y
# EVOLVE-BLOCK-END
