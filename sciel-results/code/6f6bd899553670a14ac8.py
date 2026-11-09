import numpy as np
import scipy.linalg
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 290  # 10s margin
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Check for linearity and time-invariance
    # f(t, y) = A y
    # We test f(0, y1 + y2) == f(0, y1) + f(0, y2)
    # and f(t, y) == f(0, y)
    
    # Small test vectors
    v1 = np.random.rand(n)
    v2 = np.random.rand(n)
    
    f_v1 = f(0, v1)
    f_v2 = f(0, v2)
    f_sum = f(0, v1 + v2)
    
    # Check additivity
    is_linear = np.allclose(f_sum, f_v1 + f_v2, rtol=1e-5, atol=1e-8)
    
    # Check time invariance
    f_t1 = f(t1, v1)
    is_time_invariant = np.allclose(f_t1, f_v1, rtol=1e-5, atol=1e-8)
    
    if is_linear and is_time_invariant:
        # Reconstruct A
        # A[:, j] = f(0, e_j)
        A = np.empty((n, n), dtype=float)
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            A[:, j] = f(0, e_j)
            
        # Compute matrix exponential using eigenvalue decomposition
        # This is more stable for diagonalizable matrices
        dt = t1 - t0
        A_dt = A * dt
        
        try:
            # Use eigendecomposition for better numerical stability
            eigenvalues, eigenvectors = np.linalg.eig(A_dt)
            
            # Check if matrix is diagonalizable (well-conditioned eigenvector matrix)
            cond = np.linalg.cond(eigenvectors)
            
            if cond < 1e10:  # Well-conditioned
                # exp(A) = V * diag(exp(eigenvalues)) * V^{-1}
                exp_eigenvalues = np.exp(eigenvalues)
                M = eigenvectors @ np.diag(exp_eigenvalues) @ np.linalg.inv(eigenvectors)
            else:
                # Fall back to scipy's expm if ill-conditioned
                M = scipy.linalg.expm(A_dt)
        except np.linalg.LinAlgError:
            # Fall back to scipy's expm
            M = scipy.linalg.expm(A_dt)
        
        y_final = M @ y0
        
        # Sanity check: if y_final is not finite, fallback
        if not np.all(np.isfinite(y_final)):
             return _fallback(f, t_span, y0)
             
        return y_final
    else:
        return _fallback(f, t_span, y0)

def _fallback(f, t_span, y0):
    # Use a stiff solver if linear assumption fails
    # Radau is good for stiff problems
    import scipy.integrate
    sol = scipy.integrate.solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12, max_step=(t_span[1]-t_span[0])/10)
    if sol.success:
        return sol.y[:, -1]
    else:
        # Last resort: explicit RK4 with many steps
        return _rk4(f, t_span[0], t_span[1], y0, 5000)

def _rk4(f, t0, t1, y0, steps):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            return y
    return y
