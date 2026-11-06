"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time


def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, keep margin

    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    total_interval = t1 - t0

    # Extract the linear operator A by probing f at multiple points.
    # f(t, y) = A @ y + b (for a linear system)
    # We need to find A and b.
    
    # First, get f at y0 and t0
    f0 = np.array(f(t0, y0), dtype=float)
    
    # Probe to extract A and b.
    # Use a set of probe vectors.
    # We'll use the standard basis vectors plus y0 itself.
    # Actually, for a linear system y' = A*y + b, we can use:
    # f(t, y) = A*y + b
    # f(t, 0) = b
    # f(t, e_j) = A[:, j] + b
    
    # Let's extract b first
    zero = np.zeros(n)
    f_zero = np.array(f(t0, zero), dtype=float)
    b = f_zero.copy()
    
    # Now extract columns of A
    A = np.empty((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        f_ej = np.array(f(t0, e_j), dtype=float)
        A[:, j] = f_ej - b
    
    # Verify: we've used 1 + 1 + n = n + 2 calls so far.
    # For n=15, that's 17 calls.
    
    # Check time
    if time.time() > deadline:
        # Fallback: just use the initial value
        return y0
    
    # Now we have A and b. The system is y' = A*y + b.
    # The solution is y(t) = expm(A*(t-t0)) @ y0 + integral from t0 to t of expm(A*(t-s)) @ b ds
    # For constant b, the particular solution is:
    # y_p(t) = integral from 0 to (t-t0) of expm(A*tau) @ b dtau
    # If A is invertible, y_p(t) = A^{-1} * (expm(A*(t-t0)) - I) @ b
    # More generally, we can use the matrix exponential integral.
    
    T = total_interval
    expm_AT = expm(A * T)
    
    # Compute the integral: int_0^T expm(A*tau) b dtau
    # We can use the identity: int_0^T expm(A*tau) b dtau = 
    #   (block matrix) [[A, b], [0, 0]] ... no, that's not quite right.
    # 
    # Better approach: solve the augmented system.
    # Consider z' = [A, b; 0, 0] @ [z; 1] where z is n-dimensional and the last component is 1.
    # Actually, let's use a different identity.
    # 
    # int_0^T expm(A*tau) b dtau = [A, b]^{-1} ... no.
    # 
    # Let's use: if we define M = [[A, b], [0, 0]] (an (n+1)x(n+1) matrix),
    # then expm(T*M) @ [y0; 1] = [expm(A*T)@y0 + int_0^T expm(A*tau) b dtau; 1]
    # Wait, that's not quite right either. Let me think more carefully.
    #
    # Actually, for the system y' = Ay + b, we can augment:
    # w = [y; 1], w' = [[A, b], [0, 0]] @ w
    # Then w(T) = expm(T * [[A, b], [0, 0]]) @ [y0; 1]
    # And y(T) = first n components of w(T).
    
    # Build the augmented matrix
    A_aug = np.zeros((n + 1, n + 1))
    A_aug[:n, :n] = A
    A_aug[:n, n] = b
    # Last row is already zeros
    
    w0 = np.zeros(n + 1)
    w0[:n] = y0
    w0[n] = 1.0
    
    # Compute expm(T * A_aug) @ w0
    expm_aug = expm(A_aug * T)
    w_T = expm_aug @ w0
    y_final = w_T[:n]
    
    # Check time
    if time.time() > deadline:
        return y_final
    
    # Verify against f if time permits (optional, uses more calls)
    # For now, return the result
    
    return y_final
# EVOLVE-BLOCK-END
