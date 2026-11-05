import numpy as np
import scipy.linalg as sla
import time

def integrate(f, t_span, y0):
    """
    Solves y' = f(t, y) for a 15-dimensional stiff linear system.
    Strategy: Estimate the linear operator A and offset b. If the system is
    approximately time-invariant (constant A and b), use the exact matrix
    exponential solution. This minimizes function calls (score term).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 20s margin for 300s limit

    # 1. Estimate A and b at t0 and t1 to check for time-invariance
    # We assume f(t, y) = A(t)y + b(t)
    
    # Get b(t0) and b(t1)
    b0 = f(t0, np.zeros(n))
    b1 = f(t1, np.zeros(n))
    
    # Get A(t0) columns
    A0 = np.empty((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        # f(t, e_j) = A e_j + b = A[:, j] + b
        # So A[:, j] = f(t, e_j) - b
        A0[:, j] = f(t0, e_j) - b0
        
    # Get A(t1) columns
    A1 = np.empty((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A1[:, j] = f(t1, e_j) - b1

    # Check if time-invariant
    # Use a relative tolerance
    diff_A = np.linalg.norm(A0 - A1) / (np.linalg.norm(A0) + 1e-15)
    diff_b = np.linalg.norm(b0 - b1) / (np.linalg.norm(b0) + 1e-15)
    
    T = t1 - t0
    
    if diff_A < 1e-6 and diff_b < 1e-6:
        # Constant coefficient system: y' = A y + b
        # Solution: y(T) = exp(A T) y0 + (exp(A T) - I) A^{-1} b
        # Note: If A is singular, use the series expansion for the integral part.
        
        # Compute exp(A T)
        AT = A0 * T
        exp_AT = sla.expm(AT)
        
        # Compute the convolution term: integral_0^T exp(A (T-s)) b ds
        # = (exp(A T) - I) A^{-1} b  (if A is invertible)
        # If A is singular, we can compute it as:
        # sum_{k=0}^inf (A T)^k / (k+1)! b
        # Or just solve the linear system if possible.
        
        # Try to solve A x = b
        try:
            # Check condition number
            cond = np.linalg.cond(A0)
            if cond < 1e10:
                x = np.linalg.solve(A0, b0)
                integral_term = (exp_AT - np.eye(n)) @ x
            else:
                # Use series expansion for small T or ill-conditioned A
                # term_k = (A T)^k b / (k+1)!
                term = np.zeros(n)
                current = np.zeros(n)
                current[:] = b0
                for k in range(50):
                    term += current / (k + 1)
                    current = AT @ current / (k + 1)
                    if np.linalg.norm(current / (k + 1)) < 1e-15 * np.linalg.norm(term):
                        break
                integral_term = term
        except np.linalg.LinAlgError:
            # Fallback to series
            term = np.zeros(n)
            current = np.zeros(n)
            current[:] = b0
            for k in range(50):
                term += current / (k + 1)
                current = AT @ current / (k + 1)
                if np.linalg.norm(current / (k + 1)) < 1e-15 * np.linalg.norm(term):
                    break
            integral_term = term

        y_final = exp_AT @ y0 + integral_term
        
        # Verify if time passed deadline
        if time.time() > deadline:
             return y_final
             
        return y_final
    else:
        # Time-dependent system.
        # Use a few steps of the exponential integrator method or a robust implicit solver.
        # Given the "stiff" nature, we use a small number of large steps with implicit Euler
        # or Radau. But to keep calls low, let's try 5 steps of implicit midpoint or similar.
        # Actually, for time-dependent linear y' = A(t)y + b(t), we can use the Magnus expansion
        # or just piecewise constant approximation.
        
        # Let's use 5 steps of "Exponential Integrator" with piecewise constant A.
        # y_{i+1} = exp(A_i h) y_i + (exp(A_i h) - I) A_i^{-1} b_i
        
        steps = 5
        h = T / steps
        y = y0.copy()
        t = t0
        
        for i in range(steps):
            t_mid = t + h / 2
            
            # Estimate A and b at t_mid
            b_mid = f(t_mid, np.zeros(n))
            A_mid = np.empty((n, n))
            for j in range(n):
                e_j = np.zeros(n)
                e_j[j] = 1.0
                A_mid[:, j] = f(t_mid, e_j) - b_mid
                
            AT_mid = A_mid * h
            exp_AT_mid = sla.expm(AT_mid)
            
            # Compute integral term
            try:
                cond = np.linalg.cond(A_mid)
                if cond < 1e10:
                    x = np.linalg.solve(A_mid, b_mid)
                    integral_term = (exp_AT_mid - np.eye(n)) @ x
                else:
                    term = np.zeros(n)
                    current = np.zeros(n)
                    current[:] = b_mid
                    for k in range(50):
                        term += current / (k + 1)
                        current = AT_mid @ current / (k + 1)
                        if np.linalg.norm(current / (k + 1)) < 1e-15 * np.linalg.norm(term):
                            break
                    integral_term = term
            except np.linalg.LinAlgError:
                term = np.zeros(n)
                current = np.zeros(n)
                current[:] = b_mid
                for k in range(50):
                    term += current / (k + 1)
                    current = AT_mid @ current / (k + 1)
                    if np.linalg.norm(current / (k + 1)) < 1e-15 * np.linalg.norm(term):
                        break
                integral_term = term

            y = exp_AT_mid @ y + integral_term
            t += h
            
            if time.time() > deadline:
                return y

        return y
