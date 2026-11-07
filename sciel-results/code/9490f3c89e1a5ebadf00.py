"""Gaussian RBF network with 10 fixed centers and 10 learnable amplitudes."""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: (N,) input in [-2, 2]
    params: (10,) learnable amplitudes
    Returns: (N,) predictions
    """
    # Fixed multi-scale Gaussian RBF centers and widths
    # 10 basis functions with centers spread across [-2, 2]
    # and widths from a multi-scale set
    centers = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 0.0])
    widths = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 2.0])
    
    # Compute Gaussian basis values: shape (N, 10)
    # For each point x_i and each basis k: exp(-(x_i - c_k)^2 / (2 * w_k^2))
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (N, 10)
    basis = np.exp(-0.5 * (diff / widths[np.newaxis, :]) ** 2)  # (N, 10)
    
    # Linear combination of basis functions with learnable amplitudes
    return basis @ params
