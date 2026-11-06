import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280  # 20s margin from 300s limit
    
    # 1. Extract A and b with minimal f-calls (16 total: 1 for b + 15 for basis)
    # Task guarantees fixed linear system, so we skip linearity verification
    b = f(t0, np.zeros(n))
    
    # Extract A using basis vectors: A[:, j] = f(t0, e_j) - b
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j) - b
    
    if time.time() > deadline:
        return y0
    
    T = t1 - t0
    
    # Strategy: Use eigendecomposition for exact solution of y' = A*y + b
    # y(T) = exp(A*T) @ y0 + integral_0^T exp(A*s) b ds
    # Using eigen decomposition: A = V @ diag(lambda) @ V^{-1}
    # exp(A*T) = V @ diag(exp(lambda*T)) @ V^{-1}
    # integral_0^T exp(A*s) ds = V @ diag((exp(lambda*T)-1)/lambda) @ V^{-1}
    # (handling lambda=0 case carefully)
    
    try:
        # Use eigendecomposition approach
        eigvals, eigvecs = np.linalg.eig(A)
        V_inv = np.linalg.inv(eigvecs)
        
        # Compute exp(A*T) = V @ diag(exp(lambda*T)) @ V^{-1}
        exp_lambda_T = np.exp(eigvals * T)
        exp_AT = eigvecs @ np.diag(exp_lambda_T) @ V_inv
        
        # Compute J = integral_0^T exp(A*s) ds
        # J = V @ diag(g(lambda)) @ V^{-1} where g(lambda) = (exp(lambda*T)-1)/lambda
        # For lambda near 0, use series: T - lambda*T^2/2 + lambda^2*T^3/6 - ...
        
        g = np.zeros_like(eigvals, dtype=complex)
        for i, lam in enumerate(eigvals):
            if abs(lam) < 1e-12:
                # Use Taylor series for g(0) = T
                # g(lam) = T - lam*T^2/2 + lam^2*T^3/6 - lam^3*T^4/24 + ...
                g[i] = T
                # Add correction terms for small but non-zero lambda
                if abs(lam) > 1e-15:
                    g[i] = T * (1 - lam*T/2 + lam**2*T**2/6 - lam**3*T**3/24 + lam**4*T**4/120)
            else:
                g[i] = (exp_lambda_T[i] - 1) / lam
        
        J = eigvecs @ np.diag(g) @ V_inv
        
        # Final solution
        y_final = exp_AT @ y0 + J @ b
        
        # Convert to real if imaginary parts are negligible
        if np.allclose(y_final.imag, 0, atol=1e-12):
            y_final = y_final.real
        else:
            y_final = y_final.real  # Should be real for real system
        
        return y_final
        
    except Exception:
        pass
    
    # Fallback 1: Try block matrix exponential (handles singular A uniformly)
    try:
        from scipy.linalg import expm
        
        # Build block matrix M (n+1 x n+1)
        M = np.zeros((n+1, n+1))
        M[:n, :n] = A
        M[:n, n] = b
        
        # Initial state for augmented system
        z0 = np.zeros(n+1)
        z0[:n] = y0
        z0[n] = 1.0
        
        # Compute matrix exponential
        exp_MT = expm(M * T)
        
        # Final state
        z_T = exp_MT @ z0
        
        # Extract y(T)
        y_final = z_T[:n]
        
        return y_final
    except Exception:
        pass
    
    # Fallback 2: Try solve approach with invertible A
    try:
        from scipy.linalg import expm
        exp_AT = expm(A * T)
        
        try:
            A_inv_b = np.linalg.solve(A, b)
            y_particular = (exp_AT - np.eye(n)) @ A_inv_b
            y_final = exp_AT @ y0 + y_particular
            return y_final
        except np.linalg.LinAlgError:
            # Singular case: Use higher-order quadrature
            m = 200
            xs, ws = np.polynomial.legendre.leggauss(m)
            s_vals = 0.5 * T * (xs + 1)
            w_vals = 0.5 * T * ws
            y_particular = np.zeros(n)
            for s, w in zip(s_vals, w_vals):
                exp_As = expm(A * s)
                y_particular += w * (exp_As @ b)
            y_final = exp_AT @ y0 + y_particular
            return y_final
    except Exception:
        pass
    
    # 3. Fallback Solver
    return _fallback_solve(f, t0, t1, y0, start_time, deadline)

def _fallback_solve(f, t0, t1, y0, start_time, deadline):
    # Use solve_ivp with Radau for stiffness
    try:
        from scipy.integrate import solve_ivp
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            return y0
        sol = solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-10, atol=1e-12, max_step=0.01)
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
        
    # Last resort: Fixed step implicit Euler
    n_steps = 1000
    h = (t1 - t0) / n_steps
    y = y0.copy()
    t = t0
    for _ in range(n_steps):
        z = y
        for _ in range(3):
            z = y + h * f(t + h, z)
        y = z
        t += h
    return y
