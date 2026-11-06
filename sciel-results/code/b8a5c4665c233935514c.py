"""ODE solver: 15-dimensional stiff linear system using matrix exponential."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) on [t0, t1] where f is a linear ODE.
    Strategy: extract the system matrix A and any affine term, then use
    the exact matrix exponential solution.
    
    For a linear system y' = A(t) y + b(t), if A is constant:
    y(t1) = expm((t1-t0)*A) @ y0 + integral of expm((t1-s)*A) b(s) ds
    
    For simplicity, we first try to detect if it's y' = A*y + b with constant A, b.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280  # 300s limit, keep 20s margin
    
    # Step 1: Probe the function to understand its structure
    # First, check if f is linear: f(t, y) = A(t)@y + b(t)
    # We'll probe with several vectors to extract A and b
    
    # Probe at t0 and t1 to check if A is time-dependent
    # Use a random vector to get one column of A
    # Actually, let's use basis vectors to extract A columns
    
    # Extract A at t0
    # f(t, y) = A(t) @ y + b(t)
    # Set y = 0 to get b(t)
    b_t0 = np.array(f(t0, np.zeros(n)), dtype=float)
    
    # Extract columns of A by probing with basis vectors
    A_t0 = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        # f(t0, e_j) = A_t0 @ e_j + b_t0 = A_t0[:, j] + b_t0
        A_t0[:, j] = np.array(f(t0, e_j), dtype=float) - b_t0
    
    # Check if A is time-independent by probing at midpoint
    t_mid = (t0 + t1) / 2
    b_tm = np.array(f(t_mid, np.zeros(n)), dtype=float)
    A_tm = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A_tm[:, j] = np.array(f(t_mid, e_j), dtype=float) - b_tm
    
    # Check if A is constant
    A_diff = np.max(np.abs(A_t0 - A_tm))
    b_diff = np.max(np.abs(b_t0 - b_tm))
    
    if time.time() > deadline:
        # Fallback: use a simple method
        return _fallback_rk4(f, t0, t1, y0, 1000)
    
    if A_diff < 1e-10 and b_diff < 1e-10:
        # Constant A and b: y' = A y + b
        # Solution: y(t1) = expm((t1-t0)*A) @ y0 + (expm((t1-t0)*A) - I) @ A^{-1} @ b
        # Or more stably: y(t1) = expm((t1-t0)*A) @ y0 + integral
        # For constant b: y(t1) = expm(dt*A) @ y0 + (expm(dt*A) - I) @ inv(A) @ b if A invertible
        # Better: use the formula with the augmented system or direct integration
        
        dt = t1 - t0
        M = expm(dt * A_t0)
        
        # For constant b, the particular solution is:
        # y_p(t) = (expm(t*A) - I) @ inv(A) @ b  (if A is invertible)
        # More robustly, solve the augmented system or use:
        # integral_0^dt expm(s*A) ds @ b
        
        # Compute integral_0^dt expm(s*A) ds using the formula:
        # If A is invertible: (expm(dt*A) - I) @ inv(A)
        # If A is singular, use a series or augmented matrix
        
        # Try augmented matrix approach for robustness
        # Create 2n x 2n matrix: [[A, I], [0, 0]]
        # Then expm(dt * [[A, I], [0, 0]]) = [[expm(dt*A), integral_0^dt expm(s*A) ds], [0, I]]
        
        A_aug = np.zeros((2 * n, 2 * n))
        A_aug[:n, :n] = A_t0
        A_aug[:n, n:] = np.eye(n)
        # Lower block is already zero
        
        M_aug = expm(dt * A_aug)
        M_top_left = M_aug[:n, :n]
        M_top_right = M_aug[:n, n:]
        
        y_final = M_top_left @ y0 + M_top_right @ b_t0
        
        return y_final
    
    else:
        # A or b is time-dependent; use a higher-order method
        # Use RK45 or a simple adaptive RK4 with more steps
        # Since we've already spent ~32 calls, we have budget for ~1000 more
        return _adaptive_rk4(f, t0, t1, y0, max_steps=5000)


def _fallback_rk4(f, t0, t1, y0, steps):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            break
    return y


def _adaptive_rk4(f, t0, t1, y0, max_steps=5000):
    """Fixed-step RK4 with many steps for time-dependent linear systems."""
    steps = min(max_steps, 5000)
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
