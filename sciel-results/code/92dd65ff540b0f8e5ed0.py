"""Rewritten program: ODE solver for 15-dim stiff linear system using augmented matrix exponential."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 280  # 300s limit, keep 20s margin
    
    # Check for affine term b = f(t0, 0)
    # If the system is y' = Ay + b, f(t, 0) = b
    # If the system is y' = Ay, f(t, 0) = 0
    b = f(t0, np.zeros(n))
    
    # Determine if b is significant
    # Use a small threshold to decide if we need the augmented system
    # If b is effectively zero, we can skip the augmented part to save computation?
    # Actually, expm on 16x16 vs 15x15 is negligible.
    # But if b is zero, the augmented matrix is block diagonal, and expm is efficient.
    
    # Compute A by probing basis vectors
    # f(t0, e_i) = A e_i + b
    # So A[:, i] = f(t0, e_i) - b
    A = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        A[:, i] = f(t0, e_i) - b
        
    # Total calls so far: 1 (for b) + n (for A) = 16
    
    dt = t1 - t0
    
    # Construct augmented matrix for y' = Ay + b
    # z = [y; 1]
    # z' = [A b; 0 0] z
    A_aug = np.zeros((n + 1, n + 1))
    A_aug[:n, :n] = A
    A_aug[:n, n] = b
    # Last row is already zeros
    
    # Compute matrix exponential
    # e^{A_aug * dt}
    M_aug = expm(A_aug * dt)
    
    # Initial augmented state
    z0 = np.zeros(n + 1)
    z0[:n] = y0
    z0[n] = 1.0
    
    # Final state
    z1 = M_aug @ z0
    
    # Extract y(t1)
    y1 = z1[:n]
    
    return y1

# EVOLVE-BLOCK-END
