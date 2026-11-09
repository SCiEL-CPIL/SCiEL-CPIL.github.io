"""ODE solver: 15-dimensional stiff linear system via matrix exponential."""
import numpy as np
import time


def _matrix_exp_taylor(A, t=1.0, tol=1e-16, max_terms=300):
    """Compute exp(t*A) using Taylor series with scaling-and-squaring for stability."""
    # Scaling and squaring
    # Choose p such that ||t*A|| / 2^p is small enough for Taylor to converge quickly
    norm_A = np.linalg.norm(t * A, ord=1)
    p = max(0, int(np.ceil(np.log2(norm_A / 2.0))))
    A_scaled = (t / (2 ** p)) * A
    
    # Taylor series for exp(A_scaled)
    result = np.eye(A_scaled.shape[0])
    term = np.eye(A_scaled.shape[0])
    for k in range(1, max_terms):
        term = term @ A_scaled / k
        result = result + term
        if np.linalg.norm(term, ord=1) < tol * np.linalg.norm(result, ord=1):
            break
    
    # Square p times
    for _ in range(p):
        result = result @ result
    
    return result


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Extract the system matrix A by probing with standard basis vectors.
    # For a linear system y' = A*y, f(t, y) = A*y (assuming no explicit t dependence).
    # We use f(t0, y) to get A*y for various y.
    # Probe with basis vectors: f(t0, e_i) = A*e_i = column i of A.
    # That's n calls.
    
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        e_i = np.zeros(n, dtype=float)
        e_i[i] = 1.0
        A[:, i] = f(t0, e_i)
    
    # Also check if there's a constant term: f(t0, 0)
    b = f(t0, np.zeros(n, dtype=float))
    
    # For a linear system y' = A*y + b, the solution is:
    # y(t) = exp(A*(t-t0)) * y0 + integral from t0 to t of exp(A*(t-s)) b ds
    # If b is constant: integral = (exp(A*(t-t0)) - I) * A^{-1} * b, but this requires A invertible.
    # More generally: integral = sum_{k=0}^{infty} A^k * (t-t0)^{k+1} / (k+1)! * b
    # Or: if A is invertible, integral = (exp(A*(t-t0)) - I) @ solve(A, b)
    
    dt = t1 - t0
    
    # Compute exp(A * dt)
    exp_A_dt = _matrix_exp_taylor(A, t=dt)
    
    # Compute the integral term
    # y(t1) = exp(A*dt) * y0 + integral
    # integral = sum_{k=0}^{max_terms} (A^k * dt^{k+1} / (k+1)!) * b
    
    if np.allclose(b, 0.0, atol=1e-15):
        # Purely homogeneous: y(t1) = exp(A*dt) * y0
        y_final = exp_A_dt @ y0
    else:
        # Inhomogeneous: compute integral term
        # integral = sum_{k=0}^{K} A^k * dt^{k+1} / (k+1)! * b
        integral_term = np.zeros(n, dtype=float)
        term = np.eye(n, dtype=float)  # A^0
        for k in range(0, 300):
            # term = A^k
            # contribution = term * dt^{k+1} / (k+1)! * b
            # We need dt^{k+1} / (k+1)!
            # Let's compute iteratively
            if k == 0:
                coeff = dt / 1.0  # dt^1 / 1!
            else:
                # coeff_k = dt^{k+1} / (k+1)!
                # coeff_{k-1} = dt^k / k!
                # coeff_k = coeff_{k-1} * dt / (k+1)
                coeff *= dt / (k + 1)
            
            integral_term = integral_term + coeff * (term @ b)
            
            # Check convergence
            if abs(coeff) * np.linalg.norm(b) < 1e-16 * (1 + np.linalg.norm(integral_term)):
                break
            
            term = term @ A
        
        y_final = exp_A_dt @ y0 + integral_term
    
    return y_final
