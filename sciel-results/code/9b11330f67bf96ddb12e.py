import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Model: 5 Chebyshev polynomials + 5 Gaussian RBFs.
    This combination captures global polynomial trends (via Chebyshev)
    and local non-polynomial features (via RBFs).
    """
    # 1. Chebyshev Part
    # Scale x from [-2, 2] to [-1, 1]
    x_scaled = x / 2.0
    # Clip to avoid numerical issues if x slightly out of bounds (though task says [-2,2])
    x_scaled = np.clip(x_scaled, -1.0, 1.0)
    
    T = np.zeros((5, len(x_scaled)), dtype=np.float64)
    T[0, :] = 1.0
    T[1, :] = x_scaled
    for n in range(2, 5):
        T[n, :] = 2.0 * x_scaled * T[n-1, :] - T[n-2, :]
    
    # 2. RBF Part
    # Centers: -1.5, -0.75, 0, 0.75, 1.5
    centers = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
    sigma = 0.8
    
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    G = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Combine Basis
    # params[0:5] for Chebyshev, params[5:10] for RBFs
    basis = np.hstack([T.T, G])
    
    result = basis @ params
    return result
