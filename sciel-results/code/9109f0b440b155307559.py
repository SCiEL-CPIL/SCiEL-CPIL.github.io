"""Symbolic regression for a nonlinear oscillator using a physics-informed polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# We define a basis of 10 monomials that capture common nonlinear oscillator dynamics.
# Inputs: x (position), v (velocity)
# Common terms for a Duffing-like oscillator with nonlinear damping:
# 1. 1 (bias/constant force)
# 2. x (linear restoring force)
# 3. x^3 (cubic restoring force, Duffing term)
# 4. v (linear damping)
# 5. v^3 (nonlinear damping, cubic)
# 6. x*v (parametric damping/interaction)
# 7. x^2 (quadratic nonlinearity, though often symmetric, included for completeness)
# 8. v^2 (quadratic nonlinearity in velocity)
# 9. x^5 (higher order restoring, if needed)
# 10. x^2*v (higher order interaction)

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict acceleration using a physics-informed polynomial basis.
    The function is linear in `params`, ensuring the optimization problem
    solved by the evaluator (BFGS) is convex and well-behaved.
    
    Args:
        x: Array of positions.
        v: Array of velocities.
        params: Array of 10 fitted constants.
        
    Returns:
        Array of predicted accelerations.
    """
    # Ensure x and v are 1D arrays
    x = np.asarray(x).flatten()
    v = np.asarray(v).flatten()
    
    # Construct the design matrix (N x 10)
    # Row 0: 1
    # Row 1: x
    # Row 2: x^2
    # Row 3: x^3
    # Row 4: x^5
    # Row 5: v
    # Row 6: v^2
    # Row 7: v^3
    # Row 8: x*v
    # Row 9: x^2*v
    
    X = np.column_stack((
        np.ones_like(x),
        x,
        x**2,
        x**3,
        x**5,
        v,
        v**2,
        v**3,
        x * v,
        x**2 * v
    ))
    
    # Linear combination of basis functions
    # params is (10,), X is (N, 10)
    pred = X @ params
    
    return pred
# EVOLVE-BLOCK-END
