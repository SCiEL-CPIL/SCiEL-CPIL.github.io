"""Symbolic regression for a nonlinear oscillator using physics-informed polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# We define a basis of 10 monomials that capture common nonlinear oscillator dynamics.
# Inputs: x (position), v (velocity)
# Common terms:
# 1. Linear restoring force: x
# 2. Cubic restoring force (Duffing): x^3
# 3. Linear damping: v
# 4. Velocity squared (nonlinear damping): v^2
# 5. Velocity cubed: v^3
# 6. Interaction x*v (parametric excitation/damping): x*v
# 7. Interaction x^2*v
# 8. Interaction x*v^2
# 9. Interaction x^3*v
# 10. Constant offset (bias): 1

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict acceleration using a physics-informed polynomial basis.
    The function is linear in `params`, ensuring the optimization problem
    solved by the evaluator (BFGS) is convex and well-behaved.
    
    Basis functions:
    p0: 1
    p1: x
    p2: x^2
    p3: x^3
    p4: v
    p5: v^2
    p6: v^3
    p7: x*v
    p8: x^2*v
    p9: x*v^2
    """
    # Ensure x and v are 1D arrays
    x = np.asarray(x).flatten()
    v = np.asarray(v).flatten()
    
    # Pre-compute powers for efficiency
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Construct the design matrix (N, 10)
    # Column 0: bias
    # Column 1: x
    # Column 2: x^2
    # Column 3: x^3
    # Column 4: v
    # Column 5: v^2
    # Column 6: v^3
    # Column 7: x*v
    # Column 8: x^2*v
    # Column 9: x*v^2
    
    basis = np.column_stack((
        np.ones_like(x),
        x,
        x2,
        x3,
        v,
        v2,
        v3,
        x * v,
        x2 * v,
        x * v2
    ))
    
    # Linear combination of basis functions
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
