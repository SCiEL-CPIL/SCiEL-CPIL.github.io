import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a flexible basis with multiple kernel types and
    a Chebyshev polynomial basis to capture interactions.
    
    The 10 parameters are weights for a combined basis of:
    - 6 Chebyshev-like polynomial terms (up to degree 2 with interactions)
    - 4 Gaussian RBF terms at specific centers
    
    This hybrid approach allows the optimizer to balance local and global features.
    """
    t_start = time.time()
    deadline = 15.0  # seconds, well within 120s limit
    
    N = len(x)
    
    # Basis 1-6: Polynomial/Chebyshev-like terms
    # Using normalized coordinates to [0,1] for better conditioning
    xn = (x - 0.5) / 1.5  # [0, 1]
    yn = (y - 0.5) / 1.5  # [0, 1]
    zn = (z - 0.5) / 1.5  # [0, 1]
    
    # Chebyshev polynomials of first kind
    T0 = np.ones_like(xn)
    T1_x = xn
    T1_y = yn
    T1_z = zn
    T2_x = 2 * xn**2 - 1
    T2_y = 2 * yn**2 - 1
    T2_z = 2 * zn**2 - 1
    
    # Interaction terms
    Txy = xn * yn
    Txz = xn * zn
    Tyz = yn * zn
    
    # Select 6 polynomial basis functions
    # 1. Constant
    # 2. x
    # 3. y
    # 4. z
    # 5. xy interaction
    # 6. xz interaction
    poly_basis = np.column_stack([
        T0,
        T1_x,
        T1_y,
        T1_z,
        Txy,
        Txz
    ])  # Shape (N, 6)
    
    # Basis 7-10: Gaussian RBF terms
    centers = np.array([
        [0.5, 0.5, 0.5],
        [2.0, 2.0, 2.0],
        [1.25, 1.25, 1.25],
        [0.5, 2.0, 0.5]
    ])
    
    # Adaptive width based on data spread
    X = np.column_stack((x, y, z))
    
    # Compute distances to centers
    # (N, 1, 3) - (1, 4, 3)
    D2 = np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)  # (N, 4)
    
    # Use a moderate width
    sigma = 0.5
    rbf_basis = np.exp(-D2 / (2.0 * sigma * sigma))  # (N, 4)
    
    # Combine all basis functions: (N, 10)
    full_basis = np.hstack([poly_basis, rbf_basis])
    
    # Linear combination with params
    pred = full_basis @ params
    
    return pred
