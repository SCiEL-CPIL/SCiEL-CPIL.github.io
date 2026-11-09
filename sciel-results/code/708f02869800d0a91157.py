"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Structural Change: Hybrid Basis (Global Polynomial + Local RBFs)
    # This allows capturing both global trends and local variations.
    
    x = np.asarray(x, dtype=np.float64)
    
    # Global Component: Polynomial of degree 2 (3 params)
    # y_global = p0 * 1 + p1 * x + p2 * x^2
    global_basis = np.vstack([
        np.ones_like(x),
        x,
        x**2
    ]) # Shape (3, N)
    
    # Local Component: Radial Basis Functions (Gaussian)
    # Fixed centers in the domain [0.1, 6.0]
    centers = np.array([0.2, 1.0, 2.0, 3.0, 4.0, 5.0, 5.8])
    sigma = 1.0 # Width parameter
    
    # Compute RBFs: exp(-((x - c)^2) / (2 * sigma^2))
    # x is (N,), centers is (7,)
    # diff is (N, 7)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_basis = np.exp(-0.5 * (diff / sigma)**2) # Shape (N, 7)
    
    # Combine bases:
    # global_basis is (3, N) -> transpose to (N, 3)
    # rbf_basis is (N, 7)
    # Total basis matrix B is (N, 10)
    B = np.hstack([global_basis.T, rbf_basis])
    
    # params is (10,)
    # Prediction is B @ params
    return B @ params
# EVOLVE-BLOCK-END
