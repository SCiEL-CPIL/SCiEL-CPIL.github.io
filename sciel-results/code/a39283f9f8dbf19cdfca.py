"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# Normalization constants derived from training ranges
# x: [-0.5728, 0.5275]
# v: [-0.1754, 0.2821]
X_MIN = -0.5728
X_MAX = 0.5275
V_MIN = -0.1754
V_MAX = 0.2821

X_MID = (X_MIN + X_MAX) / 2.0
X_HALF = (X_MAX - X_MIN) / 2.0
V_MID = (V_MIN + V_MAX) / 2.0
V_HALF = (V_MAX - V_MIN) / 2.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form NumPy expression for a nonlinear oscillator.
    Uses a physics-informed basis of smooth, bounded nonlinearities:
    sin(x), tanh(x), sin(v), tanh(v) and their interactions.
    This naturally captures periodic restoring forces and saturation
    while avoiding ill-conditioning of high-degree polynomials.
    """
    # Normalize inputs to [-1, 1] range for better conditioning
    xn = (x - X_MID) / X_HALF
    vn = (v - V_MID) / V_HALF
    
    # Smooth bounded nonlinearities
    sin_x = np.sin(np.pi * xn)
    tanh_x = np.tanh(2.0 * xn)
    sin_v = np.sin(np.pi * vn)
    tanh_v = np.tanh(2.0 * vn)
    
    # Also include the raw normalized values for linear terms
    # and some cross-interaction terms
    
    # Basis columns (10 total):
    # 1. Bias
    # 2. Linear x (normalized)
    # 3. sin(x) - periodic restoring force
    # 4. tanh(x) - saturation in restoring force
    # 5. Linear v (normalized) - damping
    # 6. sin(v) - periodic component in velocity
    # 7. tanh(v) - saturation in velocity effects
    # 8. x * v interaction
    # 9. x * tanh(v) - nonlinear damping
    # 10. tanh(x) * v - velocity-dependent restoring
    
    basis = np.column_stack((
        np.ones_like(x),
        xn,
        sin_x,
        tanh_x,
        vn,
        sin_v,
        tanh_v,
        xn * vn,
        xn * tanh_v,
        tanh_x * vn
    ))
    
    # Compute prediction
    return basis @ params
# EVOLVE-BLOCK-END
