"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# Fixed RBF centers and width
_RBF_CENTERS = np.linspace(-3.0, 3.0, 10)
_RBF_WIDTH = 1.0  # Standard deviation of the Gaussian


def _rbf_basis(x: np.ndarray) -> np.ndarray:
    """
    Compute RBF basis functions with fixed centers and width.
    Returns shape (n_points, 10).
    """
    n = len(x)
    basis = np.zeros((n, 10))
    for i in range(10):
        c = _RBF_CENTERS[i]
        basis[:, i] = np.exp(-((x - c) ** 2) / (2.0 * _RBF_WIDTH ** 2))
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fit a fixed-width RBF model to the data using the 10 params.
    The evaluator uses BFGS to fit params; we just need to provide the
    forward pass.
    
    params[0..9] are the weights for the 10 RBFs.
    """
    # Compute RBF basis (10 terms)
    basis = _rbf_basis(x)
    
    # Linear combination
    return basis @ params[:10]
# EVOLVE-BLOCK-END
