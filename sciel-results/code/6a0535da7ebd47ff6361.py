"""Symbolic regression with a rich, well-conditioned basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 diverse, smooth basis functions.
    The evaluator fits the 10 params via BFGS.
    """
    x = np.asarray(x, dtype=np.float64)

    # Build a diverse basis spanning polynomials, trig, and non-linear activations.
    # We avoid very high-degree polynomials or ill-scaled functions that cause
    # numerical issues with BFGS starting from all-ones.
    b = np.empty((x.size, 10), dtype=np.float64)

    # 0: constant
    b[:, 0] = 1.0
    # 1: linear
    b[:, 1] = x
    # 2: quadratic
    b[:, 2] = x * x
    # 3: cubic
    b[:, 3] = x * x * x
    # 4: quartic (scaled to avoid large values)
    b[:, 4] = (x * x * x * x) * 0.1
    # 5: sin(x)
    b[:, 5] = np.sin(x)
    # 6: cos(x)
    b[:, 6] = np.cos(x)
    # 7: sin(2x)
    b[:, 7] = np.sin(2.0 * x)
    # 8: cos(2x)
    b[:, 8] = np.cos(2.0 * x)
    # 9: tanh(x) - smooth sigmoid-like
    b[:, 9] = np.tanh(x)

    # Return the linear combination
    return b @ params
# EVOLVE-BLOCK-END
