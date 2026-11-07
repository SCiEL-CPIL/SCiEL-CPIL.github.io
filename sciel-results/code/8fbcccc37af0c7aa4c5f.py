"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_basis(x, y):
    """Build a rich basis of features from x and y."""
    basis = [
        np.ones_like(x),
        x,
        y,
        x * y,
        x**2,
        y**2,
        x**2 * y,
        x * y**2,
        x**3,
        y**3,
        np.sin(x),
        np.sin(y),
        np.cos(x),
        np.cos(y),
        np.sin(x + y),
        np.sin(x - y),
        np.exp(x),
        np.exp(y),
        np.tanh(x),
        np.tanh(y),
        np.abs(x),
        np.abs(y),
        np.sqrt(np.abs(x) + 1e-8),
        np.sqrt(np.abs(y) + 1e-8),
        np.log(np.abs(x) + 1e-8),
        np.log(np.abs(y) + 1e-8),
    ]
    return basis


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs.
    
    Uses a rich basis of features and fits parameters using the provided params array.
    The params array has length 10, so we use a subset of the basis features
    that are most likely to capture the underlying function structure.
    """
    # We have 10 parameters. Let's use a carefully chosen subset of basis features.
    # The evaluator fits params with BFGS starting from all ones.
    # We need to return a prediction that can be well-fit by 10 parameters.
    
    # Since the evaluator controls the fitting, we just need to provide a good
    # functional form. Let's use a combination that covers common patterns:
    # constant, x, y, x*y, x^2, y^2, sin(x), sin(y), cos(x), cos(y)
    
    pred = (
        params[0] * np.ones_like(x) +
        params[1] * x +
        params[2] * y +
        params[3] * x * y +
        params[4] * x**2 +
        params[5] * y**2 +
        params[6] * np.sin(x) +
        params[7] * np.sin(y) +
        params[8] * np.cos(x) +
        params[9] * np.cos(y)
    )
    
    return pred
# EVOLVE-BLOCK-END
