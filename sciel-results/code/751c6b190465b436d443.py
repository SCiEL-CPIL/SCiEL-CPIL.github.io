"""ODE solver: 15-dimensional stiff linear system.
Strategy: 
1. Extract the system matrix A and constant vector b (if affine) using 16 calls.
   - 15 calls to probe basis vectors for A.
   - 1 call to probe zero vector for b.
2. Solve the potentially affine linear ODE y' = A y + b by augmenting the system to 16 dimensions.
   z' = [A, b; 0, 0] z, where z = [y; 1].
3. Use matrix exponential on the 16x16 matrix.
Total f-calls: 16.
"""
import numpy as np
from scipy.linalg import expm
import time


def integrate(f, t_span, y0):
    """
    Integrate the 15-dimensional stiff linear ODE y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy:
    1. Extract A and b assuming f(t, y) = A y + b.
    2. Solve via augmented matrix exponential.
    
    Total f calls: 16 (15 for A, 1 for b).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # 1. Extract A using basis vectors
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
        
    # Check deadline
    if time.time() > deadline:
        return y0
        
    # 2. Extract b using zero vector
    b = f(t0, np.zeros(n, dtype=float))
    
    # Check deadline
    if time.time() > deadline:
        return y0
        
    # 3. Construct augmented system
    # z = [y; 1]
    # z' = [A, b; 0, 0] z
    B = np.zeros((n + 1, n + 1), dtype=float)
    B[:n, :n] = A
    B[:n, n] = b
    # Last row is all zeros
    
    dt = t1 - t0
    if dt == 0:
        return y0.copy()
        
    # 4. Compute matrix exponential
    E = expm(dt * B)
    
    # 5. Apply to initial state
    z0 = np.zeros(n + 1, dtype=float)
    z0[:n] = y0
    z0[n] = 1.0
    
    z_end = E @ z0
    
    # 6. Return y component
    y_final = z_end[:n]
    
    # Sanity check
    if not np.all(np.isfinite(y_final)):
        # Fallback: if expm failed, try simple RK4 on the affine system
        h = dt / 1000
        y = y0.copy()
        for _ in range(1000):
            k1 = A @ y + b
            k2 = A @ (y + 0.5 * h * k1) + b
            k3 = A @ (y + 0.5 * h * k2) + b
            k4 = A @ (y + h * k3) + b
            y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        y_final = y
        
    return y_final
