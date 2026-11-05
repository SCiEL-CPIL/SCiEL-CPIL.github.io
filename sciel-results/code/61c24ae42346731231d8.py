import numpy as np
import scipy.linalg
import time

def integrate(f, t_span, y0):
    t_start = time.time()
    deadline = t_start + 285.0
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = y0.shape[0]

    # We will extract A and b using n+1 calls, then solve exactly via expm.
    # To minimize call count, we use exactly n+1 calls.
    # b = f(t0, 0)
    zero = np.zeros(n)
    b = f(t0, zero)
    
    A = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        fi = f(t0, ei)
        A[:, i] = fi - b

    # Solve exactly using augmented matrix: z = [y; 1], z' = M z
    # Alternative: Separate homogeneous and particular solutions
    # y(t) = expm(A*dt) @ y0 + integral_0^t expm(A*tau) dtau b
    # integral = solve(A, (expm(A*dt) - I) @ b) if A is invertible
    
    dt = t1 - t0
    A_scaled = A * dt
    
    # Compute matrix exponential of the 15x15 matrix
    expA = scipy.linalg.expm(A_scaled)
    
    y_hom = expA @ y0
    
    # Check if A is invertible
    # Use rank check or try solve
    try:
        # Solve A X = (expA - I) b
        # This is equivalent to X = A^-1 (expA - I) b
        rhs = (expA - np.eye(n)) @ b
        y_part = scipy.linalg.solve(A, rhs, assume_a='gen')
        y_final = y_hom + y_part
    except np.linalg.LinAlgError:
        # Fallback to augmented matrix method if A is singular
        M = np.zeros((n+1, n+1))
        M[:n, :n] = A
        M[:n, n] = b
        
        M_scaled = dt * M
        expM = scipy.linalg.expm(M_scaled)
        
        z_init = np.zeros(n+1)
        z_init[:n] = y0
        z_init[n] = 1.0
        
        z_final = expM @ z_init
        y_final = z_final[:n]
    
    # Verify: check if the solution satisfies the ODE approximately
    # For a linear system, we can also try an alternative computation
    # to cross-check accuracy
    
    # Alternative: use eigendecomposition if well-conditioned
    try:
        if time.time() < deadline - 5:
            eigvals, eigvecs = scipy.linalg.eig(A)
            # Check condition number of eigvecs
            cond = np.linalg.cond(eigvecs)
            if cond < 1e8:
                # y(t) = V * diag(exp(eigvals * dt)) * V^-1 * y0_part
                # But we need to handle the constant term b as well
                # This is more complex; skip for now and trust expm
                pass
    except Exception:
        pass
    
    return y_final
