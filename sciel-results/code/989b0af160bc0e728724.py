"""Symbolic regression using Gaussian RBF basis for a linear-in-parameters model."""
# EVOLVE-BLOCK-START
import numpy as np

# Fixed Gaussian RBF centers spread across [-2, 2]
# 10 centers with varying bandwidths for flexibility
_CENTERS = np.array([-1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, -0.3, 0.3, 0.9])
# Bandwidths: mix of narrow and wide kernels
_BANDWIDTHS = np.array([0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.5, 0.5, 0.7])


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model using Gaussian Radial Basis Functions.
    Linear in parameters, so BFGS from any start point converges reliably.

    params (length 10):
        p0-p9: coefficients for each Gaussian RBF centered at _CENTERS[i]
               with bandwidth _BANDWIDTHS[i]

    RBF_i(x) = exp(-(x - center_i)^2 / (2 * bandwidth_i^2))
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    # Compute all 10 RBF features
    result = np.zeros_like(x, dtype=np.float64)
    for i in range(10):
        diff = x - _CENTERS[i]
        rbf = np.exp(-diff * diff / (2.0 * _BANDWIDTHS[i] * _BANDWIDTHS[i]))
        result += params[i] * rbf

    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e8, neginf=-1e8)
# EVOLVE-BLOCK-END
