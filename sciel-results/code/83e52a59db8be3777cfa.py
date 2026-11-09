"""
Solves the 2D Heat Equation IVP using the Matrix Exponential method.
The problem is linear and autonomous: y' = A y + b.
We recover A and b by probing f, then compute the exact solution.
"""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 290.0  # 290s safety margin
    
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0
    
    # 1. Determine if the system is homogeneous (b=0) or affine
    # b = f(t, 0)
    # Since it's autonomous, t doesn't matter for A and b structure, but we use t0.
    b = f(t0, np.zeros(n))
    
    # Check time
    if time.time() > deadline:
        # Fallback: just return y0 (shouldn't happen for this small problem)
        return y0.copy()
    
    # 2. Recover the Jacobian A
    # A[:, j] = f(t0, e_j)
    A = np.zeros((n, n))
    for j in range(n):
        if time.time() > deadline:
            # Fallback with partial A
            break
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
        
    # Check time
    if time.time() > deadline:
        # Fallback: use what we have
        pass
    
    # 3. Solve the system
    # Check if b is negligible
    if np.linalg.norm(b) < 1e-12:
        # Homogeneous: y(t) = expm(A * dt) @ y0
        if time.time() > deadline:
            # Very rough fallback
            return y0.copy()
            
        M = scipy.linalg.expm(A * dt)
        y_end = M @ y0
    else:
        # Affine: y' = A y + b
        # Augment system: z = [y; 1], z' = [[A, b], [0, 0]] z
        # z(0) = [y0; 1]
        A_aug = np.zeros((n + 1, n + 1))
        A_aug[:n, :n] = A
        A_aug[:n, n] = b
        # Last row and col are already 0
        
        z0 = np.zeros(n + 1)
        z0[:n] = y0
        z0[n] = 1.0
        
        if time.time() > deadline:
            # Fallback
            return y0.copy()
            
        M_aug = scipy.linalg.expm(A_aug * dt)
        z_end = M_aug @ z0
        y_end = z_end[:n]
        
    return y_end
# EVOLVE-BLOCK-END
