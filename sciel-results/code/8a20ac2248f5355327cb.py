"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Basis functions: [1, x, x^2, x^3, x^4, sin(x), cos(x), exp(x), |x|, tanh(x)]
    # This is a richer set of basis functions than just linear, allowing the
    # optimizer to fit more complex functions.
    x = np.asarray(x)
    basis = np.column_stack([
        np.ones_like(x),
        x,
        x ** 2,
        x ** 3,
        x ** 4,
        np.sin(x),
        np.cos(x),
        np.exp(x),
        np.abs(x),
        np.tanh(x)
    ])
    return basis @ params
# EVOLVE-BLOCK-END
