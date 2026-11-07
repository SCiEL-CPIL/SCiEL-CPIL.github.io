"""Symbolic regression: equation for a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form acceleration for a nonlinear oscillator.
    
    Uses a physics-informed basis of smooth, bounded nonlinearities.
    The basis includes:
    1.  1
    2.  sin(x)       - primary restoring force (linear for small x)
    3.  x            - linear restoring (redundant with sin(x) for small x, but helps fit)
    4.  v            - linear damping
    5.  v * sin(x)   - state-dependent damping (nonlinear oscillator feature)
    6.  x * sin(x)   - nonlinear stiffness correction
    7.  sin(2*x)     - higher harmonic restoring force
    8.  v * v * sin(x) - higher order state-dependent damping
    9.  x * x * sin(x) - higher order nonlinear stiffness
    10. sin(x) * sin(x) * sin(x) - cubic nonlinearity in restoring force
    
    This basis naturally captures the periodic nature of oscillators and
    avoids the ill-conditioning of high-degree polynomials.
    """
    # Normalize inputs to help with conditioning, though sin/tanh are bounded
    # x range: [-0.5728, 0.5275]
    # v range: [-0.1754, 0.2821]
    
    # Use moderate scaling to keep arguments in a range where sin is well-conditioned
    # but not so large that we lose resolution. Since sin is periodic, scaling
    # changes the effective frequency. For a standard oscillator, x is the phase.
    # We keep x roughly as-is or slightly scaled.
    
    x_s = x
    v_s = v
    
    # Basis functions
    b0 = np.ones_like(x_s)
    b1 = np.sin(x_s)
    b2 = x_s
    b3 = v_s
    b4 = v_s * np.sin(x_s)
    b5 = x_s * np.sin(x_s)
    b6 = np.sin(2.0 * x_s)
    b7 = v_s * v_s * np.sin(x_s)
    b8 = x_s * x_s * np.sin(x_s)
    b9 = np.sin(x_s) * np.sin(x_s) * np.sin(x_s)
    
    # Linear combination
    return (params[0] * b0 + params[1] * b1 + params[2] * b2 +
            params[3] * b3 + params[4] * b4 + params[5] * b5 +
            params[6] * b6 + params[7] * b7 + params[8] * b8 +
            params[9] * b9)
# EVOLVE-BLOCK-END
