"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis model: 5 Chebyshev polynomials (T0..T4) + 5 local Gaussian RBFs.
    
    The Chebyshev terms provide a well-conditioned global polynomial trend,
    while the Gaussian RBFs add local flexibility to capture non-polynomial
    features (sharper transitions, localized bumps) that a pure degree-9
    polynomial might struggle with or require high-degree terms to approximate.
    
    This hybrid approach leverages the stability of Chebyshev for the global
    shape and the locality of RBFs for fine-grained details, potentially
    achieving a lower MSE than either basis alone.
    
    Design:
      - T_0, T_1, T_2, T_3, T_4 on t in [-1, 1]
      - 5 Gaussian RBFs centered at 1.0, 2.0, 3.0, 4.0, 5.0 with sigma=0.8
    
    Total: 10 basis functions, 10 parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure finite inputs
    x = np.where(np.isfinite(x), x, 3.0)
    p = np.where(np.isfinite(p), p, 0.0)
    
    # Scale x from [0.1, 6] to t in [-1, 1] for Chebyshev
    # Midpoint = 3.05, Half-width = 2.95
    t = (x - 3.05) / 2.95
    t = np.clip(t, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0 through T_4
    T0 = np.ones_like(t)
    T1 = t.copy()
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    
    # Compute Gaussian RBFs
    centers = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sigma = 0.8
    
    # diff is (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    RBF = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    
    # Stack basis (N, 10)
    # Columns: T0, T1, T2, T3, T4, RBF0, RBF1, RBF2, RBF3, RBF4
    X = np.hstack([
        T0[:, np.newaxis],
        T1[:, np.newaxis],
        T2[:, np.newaxis],
        T3[:, np.newaxis],
        T4[:, np.newaxis],
        RBF
    ])
    
    return X @ p
# EVOLVE-BLOCK-END
