"""Symbolic regression: Gaussian RBF basis with 10 parameters."""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 10 fixed Gaussian RBF centers spread across [-2, 2]
    # Using distinct, well-separated centers to avoid near-collinearity
    centers = np.array([
        -2.0, -1.5, -1.0, -0.5, 0.0,
        0.5, 1.0, 1.5, 2.0, 0.0
    ])
    
    # Widths: use a moderate width so each RBF has local influence
    # but there's enough overlap for smooth interpolation
    # Using width ~0.8 gives good coverage of [-2, 2] with 10 centers
    width = 0.8
    
    # Compute Gaussian RBF values: exp(-((x - c)^2) / (2 * width^2))
    # x shape: (N,), centers shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-(diff ** 2) / (2.0 * width ** 2))
    
    # Linear combination: params @ rbf_values.T
    # rbf_values shape: (N, 10), params shape: (10,)
    prediction = rbf_values @ params
    
    return prediction
