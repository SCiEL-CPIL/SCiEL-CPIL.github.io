# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: global polynomial terms + local RBFs.
    First 4 params: coefficients for [1, x, x^2, x^3]
    Last 6 params: weights for RBFs at fixed centers in [0.1, 6.0]
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Global polynomial terms (first 4 params)
    poly_basis = np.stack([
        np.ones_like(x),
        x,
        x**2,
        x**3
    ], axis=1)
    
    # Local RBFs (last 6 params)
    # Fixed centers spread across [0.1, 6.0]
    centers = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    sigma = 1.0  # RBF width
    
    rbf_terms = np.zeros((len(x), 6))
    for i, c in enumerate(centers):
        rbf_terms[:, i] = np.exp(-((x - c)**2) / (2.0 * sigma**2))
    
    # Combine basis
    basis = np.hstack([poly_basis, rbf_terms])
    
    pred = basis @ params
    return pred
# EVOLVE-BLOCK-END
