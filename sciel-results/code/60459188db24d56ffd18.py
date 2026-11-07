"""RBF + polynomial symbolic regression candidate."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 5 RBF centers
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    # Gaussian RBF with fixed width
    width = 1.0
    # Compute RBF terms: exp(-width * (x - center)^2)
    # x shape: (N,), centers shape: (5,)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (N, 5)
    rbf = np.exp(-width * diff ** 2)  # (N, 5)
    
    # 5 polynomial terms: 1, x, x^2, x^3, x^4
    poly = np.column_stack([
        np.ones_like(x),
        x,
        x ** 2,
        x ** 3,
        x ** 4
    ])  # (N, 5)
    
    # params[0:5] for RBF, params[5:10] for polynomial
    y_rbf = rbf @ params[0:5]
    y_poly = poly @ params[5:10]
    
    return y_rbf + y_poly
# EVOLVE-BLOCK-END
