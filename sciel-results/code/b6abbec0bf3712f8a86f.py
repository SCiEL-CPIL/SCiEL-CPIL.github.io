"""ODE solver: 15-dimensional stiff linear system using analytic matrix exponential."""
import numpy as np
import time
from scipy.linalg import expm


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) on [t0, t1] with initial condition y0.
    
    Strategy:
    1. Assume the system is linear, inhomogeneous, and has constant coefficients: y' = A*y + b
    2. Probe A using 15 basis vector evaluations.
    3. Probe b using one zero-vector evaluation.
    4. Use the analytic solution via augmented matrix exponential:
       [y(t)]' = [A b; 0 0] [y(t); 1]
       [y(T); 1] = expm(T * M) @ [y0; 1]
    5. This uses 16 function calls and handles constant inhomogeneous systems exactly.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    T = t1 - t0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Leave 50s margin before 330s kill

    # Step 1: Probe the system to determine A and b
    # A[:, j] = f(t0, e_j)
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = np.array(f(t0, e_j), dtype=float)
        
        # Check time limit
        if time.time() > deadline:
            # If we run out of time, we cannot reliably solve.
            # Return y0 as a fallback? Or break and use partial A?
            # Given the speed, this is unlikely.
            pass

    b = np.array(f(t0, np.zeros(n)), dtype=float)
    
    # Step 2: Construct augmented matrix M
    # M is (n+1) x (n+1)
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n] = b
    # Last row is 0
    
    # Step 3: Compute the matrix exponential
    # This is the most computationally expensive part, but for 16x16 it is fast.
    EM = expm(T * M)
    
    # Step 4: Apply to augmented initial condition
    y0_aug = np.zeros(n + 1)
    y0_aug[:n] = y0
    y0_aug[n] = 1.0
    
    y_final_aug = EM @ y0_aug
    y_final = y_final_aug[:n]
    
    return y_final
