"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Uses a scaled cubic polynomial basis to improve numerical conditioning.
    Inputs are scaled to be roughly in [-1, 1] to help BFGS convergence.
    """
    # Scaling factors based on training data ranges
    # x in [-0.5728, 0.5275] -> scale by 0.6
    # v in [-0.1754, 0.2821] -> scale by 0.3
    xs = x / 0.6
    vs = v / 0.3
    
    # Precompute powers
    xs2 = xs * xs
    vs2 = vs * vs
    xs3 = xs2 * xs
    vs3 = vs2 * vs
    
    # Basis terms (10 terms)
    # Order chosen to cover linear, quadratic, and cubic interactions
    term0 = xs          # Linear x
    term1 = vs          # Linear v
    term2 = xs2         # Quadratic x
    term3 = vs2         # Quadratic v
    term4 = xs * vs     # Cross quadratic
    term5 = xs3         # Cubic x
    term6 = vs3         # Cubic v
    term7 = xs2 * vs    # Cross cubic (Van der Pol style)
    term8 = xs * vs2    # Cross cubic
    term9 = np.ones_like(x) # Constant
    
    # Linear combination
    return (params[0] * term0 +
            params[1] * term1 +
            params[2] * term2 +
            params[3] * term3 +
            params[4] * term4 +
            params[5] * term5 +
            params[6] * term6 +
            params[7] * term7 +
            params[8] * term8 +
            params[9] * term9)
# EVOLVE-BLOCK-END
