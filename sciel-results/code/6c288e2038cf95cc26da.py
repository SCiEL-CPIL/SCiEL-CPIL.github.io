import numpy as np
import scipy.linalg
import time

def integrate(f, t_span, y0):
    t_start = time.time()
    deadline = t_start + 280.0  # Safety margin before 300s limit
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = y0.shape[0]

    # Extract A and b using n+1 calls
    zero = np.zeros(n)
    b = f(t0, zero)
    
    A = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        fi = f(t0, ei)
        A[:, i] = fi - b

    dt = t1 - t0
    A_scaled = A * dt

    # Compute matrix exponential using eigendecomposition for better control
    # y(t) = exp(A*dt) @ y0 + integral_0^t exp(A*tau) b dtau
    #      = exp(A*dt) @ y0 + A^{-1} (exp(A*dt) - I) b  [if A invertible]
    
    # Use eigendecomposition: A = V diag(lambda) V^{-1}
    # exp(A*dt) = V diag(exp(lambda*dt)) V^{-1}
    # integral = V diag((exp(lambda*dt) - 1)/lambda) V^{-1} b
    
    try:
        eigvals, eigvecs = scipy.linalg.eig(A)
        eigvecs_inv = scipy.linalg.inv(eigvecs)
        
        exp_eigvals = np.exp(eigvals * dt)
        
        # exp(A*dt) = V @ diag(exp_eigvals) @ V^{-1}
        expA = eigvecs @ np.diag(exp_eigvals) @ eigvecs_inv
        
        # For the particular solution:
        # integral_0^t exp(A*tau) b dtau = V @ diag((exp(lambda*dt)-1)/lambda) @ V^{-1} @ b
        # Handle lambda=0 case: (exp(lambda*dt)-1)/lambda -> dt as lambda->0
        coeff = np.zeros(n, dtype=complex)
        for i in range(n):
            if abs(eigvals[i]) < 1e-14:
                # Use Taylor: (exp(x)-1)/x = 1 + x/2 + x^2/6 + ...
                x = eigvals[i] * dt
                coeff[i] = 1.0 + x/2.0 + x*x/6.0
            else:
                coeff[i] = (exp_eigvals[i] - 1.0) / eigvals[i]
        
        y_hom = expA @ y0
        y_part = eigvecs @ np.diag(coeff) @ eigvecs_inv @ b
        y_final = y_hom + y_part
    except Exception:
        # Fallback to scipy expm if eigendecomposition fails
        expA = scipy.linalg.expm(A_scaled)
        y_hom = expA @ y0
        try:
            rhs = (expA - np.eye(n)) @ b
            y_part = scipy.linalg.solve(A, rhs, assume_a='gen')
        except np.linalg.LinAlgError:
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
            return y_final
        y_final = y_hom + y_part

    # Take real part if result is complex (shouldn't be for real systems)
    if np.iscomplexobj(y_final):
        y_final = np.real(y_final)

    return y_final
