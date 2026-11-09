"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    n_points = len(x)
    if n_points == 0:
        return np.array([])
    
    # Sigmoid (logistic) basis functions with distinct centers and scales
    # Centers spread across [-2, 2], scales vary for different transition widths
    centers = np.linspace(-2.0, 2.0, 10)
    scales = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    
    # Compute sigmoid basis: phi_i(x) = 1 / (1 + exp(-scale_i * (x - center_i)))
    # This is strictly linear in params, so BFGS converges to global LS optimum
    basis = np.zeros((n_points, 10))
    for i in range(10):
        z = scales[i] * (x - centers[i])
        # Numerically stable sigmoid
        basis[:, i] = np.where(z >= 0,
                               1.0 / (1.0 + np.exp(-z)),
                               np.exp(z) / (1.0 + np.exp(z)))
    
    # Linear combination
    return basis @ params
# EVOLVE-BLOCK-END
