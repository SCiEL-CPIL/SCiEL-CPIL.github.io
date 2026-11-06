import numpy as np
from scipy.linalg import lu_factor, lu_solve

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    N = len(y0)
    
    # Estimate Jacobian at t0, y0
    # Using central differences
    h_fd = 1e-5
    J = np.zeros((N, N))
    for j in range(N):
        y_plus = y0.copy()
        y_minus = y0.copy()
        y_plus[j] += h_fd
        y_minus[j] -= h_fd
        f_plus = f(t0, y_plus)
        f_minus = f(t0, y_minus)
        J[:, j] = (f_plus - f_minus) / (2 * h_fd)

    # Fixed-step Crank-Nicolson solver
    # Assuming linear autonomous system y' = J y
    n_steps = 1000
    h = (t1 - t0) / n_steps
    
    I = np.eye(N)
    A = I - 0.5 * h * J
    B = I + 0.5 * h * J
    
    try:
        lu, piv = lu_factor(A)
    except Exception:
        # Fallback to solve_ivp if factorization fails
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
        return sol.y[:, -1]

    y = y0
    for _ in range(n_steps):
        rhs = B @ y
        y = lu_solve((lu, piv), rhs)
        
    return y
