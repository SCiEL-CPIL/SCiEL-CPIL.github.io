"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a rich basis of 10 features.
    The basis spans polynomials, trigonometric functions, Gaussian,
    and saturating nonlinearities to cover a wide range of possible
    target functions.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params

    # Compute basis functions
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x

    sin_x = np.sin(x)
    cos_x = np.cos(x)
    exp_neg_x2 = np.exp(-x2)
    tanh_x = np.tanh(x)

    # 10 basis functions:
    # 1. constant
    # 2. x
    # 3. x^2
    # 4. x^3
    # 5. x^4
    # 6. sin(x)
    # 7. cos(x)
    # 8. exp(-x^2)
    # 9. tanh(x)
    # 10. x * tanh(x)

    basis = np.column_stack([
        np.ones_like(x),
        x,
        x2,
        x3,
        x4,
        sin_x,
        cos_x,
        exp_neg_x2,
        tanh_x,
        x * tanh_x
    ])

    return basis @ p
# EVOLVE-BLOCK-END
