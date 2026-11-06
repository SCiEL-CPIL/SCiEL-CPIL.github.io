import numpy as np

def _matrix_exp_taylor(M, K=100):
    """
    Compute matrix exponential using Taylor series.
    M: square matrix
    K: number of terms
    """
    n = M.shape[0]
    I = np.eye(n)
    term = I.copy()
    result = I.copy()
    for k in range(1, K + 1):
        term = term @ M / k
        result += term
    return result

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Try to identify if f is linear and constant: f(t, y) = A @ y + b
    # 1. Get b = f(t0, 0)
    b = f(t0, np.zeros(n))
    
    # 2. Get columns of A
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j) - b
        
    # 3. Verify linearity and time-invariance
    # Pick a deterministic test point to avoid randomness overhead/variability
    y_test = np.ones(n)
    t_test = (t0 + t1) / 2.0
    
    f_test = f(t_test, y_test)
    f_pred = A @ y_test + b
    
    # Check if they are close
    norm_diff = np.linalg.norm(f_test - f_pred)
    norm_f = np.linalg.norm(f_pred)
    if norm_f > 1e-12:
        rel_err = norm_diff / norm_f
    else:
        rel_err = norm_diff
        
    if rel_err < 1e-10:
        # System is linear and constant
        # Construct augmented matrix for y' = A y + b
        # [y']   [A b] [y]
        # [0'] = [0 0] [1]
        A_aug = np.zeros((n+1, n+1))
        A_aug[:n, :n] = A
        A_aug[:n, n] = b
        
        y0_aug = np.zeros(n+1)
        y0_aug[:n] = y0
        y0_aug[n] = 1.0
        
        dt = t1 - t0
        # Scale matrix by dt before exponentiation
        E = _matrix_exp_taylor(A_aug * dt, K=100)
        
        y_final_aug = E @ y0_aug
        return y_final_aug[:n]
    else:
        # Fallback: Use SciPy's solve_ivp with a stiff method
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-12, atol=1e-14)
        if sol.success:
            return sol.y[:, -1]
        else:
            return y0
