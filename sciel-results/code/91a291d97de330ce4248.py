"""ODE solver: 15-dimensional stiff linear system using matrix extraction."""
import numpy as np
from scipy.linalg import expm


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Extract system matrix A assuming y' = A y (homogeneous linear system).
    # We use n orthogonal vectors to determine A with exactly n calls.
    # Using an orthogonal matrix V minimizes the condition number of the linear system
    # used to recover A, thus minimizing numerical error in the extraction step.
    
    # Generate a random matrix and orthogonalize it via QR decomposition
    # This ensures V is orthogonal (V^T V = I), so V^{-1} = V^T
    R = np.random.randn(n, n)
    Q, _ = np.linalg.qr(R)
    V = Q
    
    # Probe f with the columns of V
    F = np.empty((n, n))
    for i in range(n):
        F[:, i] = f(t0, V[:, i])
        
    # Solve for A: F = A @ V  => A = F @ V^T
    # Since V is orthogonal, V^{-1} is V^T
    A = F @ V.T
    
    dt = t1 - t0
    
    # Compute the matrix exponential
    M = expm(A * dt)
    
    # Final state
    y_end = M @ y0
    
    return y_end
