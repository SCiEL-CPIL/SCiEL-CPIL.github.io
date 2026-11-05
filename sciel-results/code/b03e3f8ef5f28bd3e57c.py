"""ODE solver: 15-dimensional stiff linear system using scaling-and-squaring with Taylor series."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_jacobian(f, t, y):
    """Compute the Jacobian of f(t, y) using forward finite differences."""
    n = len(y)
    J = np.empty((n, n))
    scale = 1e-7
    for j in range(n):
        e = np.zeros(n)
        e[j] = scale * max(1.0, abs(y[j]))
        J[:, j] = (f(t, y + e) - f(t, y)) / e[j]
    return J


def _matrix_exponential_taylor(A, order=30):
    """Compute exp(A) using Taylor series with sufficient terms for accuracy."""
    n = A.shape[0]
    I = np.eye(n)
    result = I.copy()
    term = I.copy()
    for k in range(1, order + 1):
        term = term @ A / k
        result = result + term
    return result


def _scaling_squaring_exp(A, max_norm=1e6):
    """Compute exp(A) using scaling-and-squaring method with Taylor series."""
    # Compute the norm of A
    norm_A = np.linalg.norm(A, ord=np.inf)
    
    if norm_A < 1e-15:
        return np.eye(A.shape[0])
    
    # Choose s such that ||A / 2^s|| is small enough for Taylor series
    # We want ||A / 2^s|| < 1 for good convergence
    if norm_A < 1:
        s = 0
    else:
        s = int(np.ceil(np.log2(norm_A))) + 1
    
    # Scale A
    A_scaled = A / (2 ** s)
    
    # Compute exp(A_scaled) using Taylor series
    exp_scaled = _matrix_exponential_taylor(A_scaled, order=50)
    
    # Square s times
    result = exp_scaled
    for _ in range(s):
        result = result @ result
    
    return result


def integrate(f, t_span, y0):
    """
    Integrate the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    For a linear system y' = A(t) * y, we can use the matrix exponential approach.
    Since the problem is described as a "fixed" stiff linear system, we assume A is constant.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 300  # 300 seconds max
    
    # Estimate the Jacobian (assumed constant for linear system)
    # Use a point in the middle for better estimation
    t_mid = (t0 + t1) / 2
    y_mid = y.copy()
    
    # Get f at initial point
    f0 = f(t0, y)
    
    # Compute Jacobian at t0
    A = _compute_jacobian(f, t0, y)
    
    # Check if the system is truly linear by comparing f at different points
    # For a linear system: f(t, y) = A @ y (if homogeneous) or A @ y + b
    # Let's check if f(t, 0) gives the offset
    y_zero = np.zeros(n)
    f_zero = f(t0, y_zero)
    
    # If f_zero is non-zero, the system is affine: y' = A*y + f_zero
    # We can handle this by augmenting the system
    
    has_offset = np.linalg.norm(f_zero) > 1e-10
    
    if has_offset:
        # Augment the system: [y; 1]' = [A, f_zero; 0, 0] [y; 1]
        A_aug = np.zeros((n + 1, n + 1))
        A_aug[:n, :n] = A
        A_aug[:n, n] = f_zero
        y_aug = np.zeros(n + 1)
        y_aug[:n] = y
        y_aug[n] = 1.0
    else:
        A_aug = A
        y_aug = y.copy()
    
    dt = t1 - t0
    
    # Use scaling-and-squaring to compute exp(A_aug * dt)
    if dt == 0:
        result = y_aug
    else:
        A_dt = A_aug * dt
        exp_A_dt = _scaling_squaring_exp(A_dt)
        result = exp_A_dt @ y_aug
    
    # Extract the solution
    if has_offset:
        y_final = result[:n]
    else:
        y_final = result
    
    return y_final
# EVOLVE-BLOCK-END
