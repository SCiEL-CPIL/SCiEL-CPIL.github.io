"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We use a hybrid basis:
# - 4 Legendre polynomials (orthogonal on [-1,1], well-conditioned for BFGS)
# - 4 trigonometric terms (sin/cos at two frequencies) for periodic structure
# - 2 adaptive Gaussian bumps (center and width parameterized by pairs of params)
#   This allows the optimizer to place local features adaptively.
#
# Parameter layout (10 total):
#   p[0:4]  -> Legendre coefficients L0..L3
#   p[4:8]  -> sin(x), cos(x), sin(2x), cos(2x) coefficients
#   p[8], p[9] -> Gaussian bump: center = tanh(p[8])*3, width = 0.3 + 1.5*sigmoid(p[9])
#                 (tanh keeps center in [-3,3], sigmoid keeps width in [0.3, 1.8])

def _legendre(n, x):
    """Compute Legendre polynomial P_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    Pnm2 = np.ones_like(x)
    Pnm1 = x.copy()
    for k in range(1, n):
        Pn = ((2 * k + 1) * x * Pnm1 - k * Pnm2) / (k + 1)
        Pnm2 = Pnm1
        Pnm1 = Pn
    return Pnm1


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters.
    Returns: array of predictions, one per point.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params

    # Normalize x to [-1, 1] for Legendre polynomials
    u = x / 3.0

    # Legendre polynomials L0..L3 (orthogonal, well-conditioned)
    L0 = _legendre(0, u)
    L1 = _legendre(1, u)
    L2 = _legendre(2, u)
    L3 = _legendre(3, u)

    # Trigonometric terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2.0 * x)
    cos_2x = np.cos(2.0 * x)

    # Adaptive Gaussian bump:
    #   center = tanh(p[8]) * 3.0  (in [-3, 3])
    #   width  = 0.3 + 1.5 * sigmoid(p[9])  (in [0.3, 1.8])
    center = np.tanh(p[8]) * 3.0
    # sigmoid(p[9]) = 1 / (1 + exp(-p[9])), clipped for numerical stability
    sig = 1.0 / (1.0 + np.exp(-np.clip(p[9], -50, 50)))
    width = 0.3 + 1.5 * sig
    gauss = np.exp(-0.5 * ((x - center) / width) ** 2)

    # Combine all terms
    y = (p[0] * L0 +
         p[1] * L1 +
         p[2] * L2 +
         p[3] * L3 +
         p[4] * sin_x +
         p[5] * cos_x +
         p[6] * sin_2x +
         p[7] * cos_2x +
         p[8] * gauss)

    # Note: p[9] is used inside the Gaussian width computation, not as a linear coefficient.
    # We return y which already incorporates p[9] through the width.
    return y
# EVOLVE-BLOCK-END
