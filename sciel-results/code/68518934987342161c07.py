"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# The evaluator fits params (length 10) via BFGS starting from all ones.
# We want equation(x, params) to be a flexible model that, given 10 params,
# can approximate a wide range of smooth functions on [-3, 3].
#
# Strategy: Use a set of 10 basis functions that are well-scaled and diverse.
# The key insight is that BFGS needs a well-conditioned optimization landscape.
# We'll use a combination of:
# 1. Low-degree polynomials (1, x, x^2, x^3) - capture global shape
# 2. Scaled trigonometric functions (sin(x), cos(x), sin(2x), cos(2x)) - capture periodicity
# 3. Gaussian-like bumps at different locations - capture local features
#
# Important: Scale features so that they are O(1) on [-3,3] to help BFGS converge.

def _build_features(x: np.ndarray) -> np.ndarray:
    """
    Build a feature matrix of shape (n, 10) with diverse, well-scaled basis functions.
    Designed to capture polynomials up to degree 3, periodic components, and local bumps.
    All features are scaled to be O(1) on [-3, 3] for BFGS stability.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    F = np.zeros((n, 10), dtype=np.float64)
    
    # Polynomial terms (scaled for stability)
    F[:, 0] = 1.0                              # constant
    F[:, 1] = x / 3.0                          # linear, scaled to [-1, 1]
    F[:, 2] = (x / 3.0) ** 2                   # quadratic, scaled to [0, 1]
    F[:, 3] = (x / 3.0) ** 3                   # cubic, scaled to [-1, 1]
    
    # Trigonometric terms with multiple frequencies
    F[:, 4] = np.sin(x)                        # fundamental sine
    F[:, 5] = np.cos(x)                        # fundamental cosine
    F[:, 6] = np.sin(2.0 * x)                  # second harmonic sine
    F[:, 7] = np.cos(2.0 * x)                  # second harmonic cosine
    
    # Two Gaussian bumps: one at center, one offset
    # These help capture local non-smooth features
    F[:, 8] = np.exp(-0.5 * (x / 1.2) ** 2)                    # center bump
    F[:, 9] = np.exp(-0.5 * ((x - 2.0) / 1.0) ** 2)            # right-side bump
    
    return F


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form linear combination of 10 pre-defined basis functions.
    params: array of length 10, coefficients for each feature.
    Returns: array of predictions, one per point.
    """
    F = _build_features(x)
    return F @ params
# EVOLVE-BLOCK-END
