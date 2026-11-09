import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using normalized Chebyshev basis (T and U mixed).
    
    Uses 5 Chebyshev T-polynomials (T_0..T_4) and 5 Chebyshev U-polynomials (U_0..U_4),
    all column-normalized to unit L2 norm for better BFGS conditioning.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map x from [0.1, 6] to [-1, 1]
    t = (x - 0.1) * (2.0 / 5.9) - 1.0
    
    n = len(t)
    if n == 0:
        return np.array([])
    
    # Compute Chebyshev T polynomials: T_0 .. T_4
    T = np.zeros((n, 5))
    T[:, 0] = 1.0
    T[:, 1] = t
    for k in range(2, 5):
        T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
    
    # Compute Chebyshev U polynomials: U_0 .. U_4
    # U_0 = 1, U_1 = 2t, U_k = 2t*U_{k-1} - U_{k-2}
    U = np.zeros((n, 5))
    U[:, 0] = 1.0
    U[:, 1] = 2.0 * t
    for k in range(2, 5):
        U[:, k] = 2.0 * t * U[:, k-1] - U[:, k-2]
    
    # Concatenate: [T_0..T_4, U_0..U_4]
    basis = np.hstack([T, U])
    
    # Normalize each column to unit L2 norm for better conditioning
    norms = np.linalg.norm(basis, axis=0)
    norms[norms < 1e-12] = 1.0  # avoid division by zero
    basis = basis / norms
    
    return basis @ p
