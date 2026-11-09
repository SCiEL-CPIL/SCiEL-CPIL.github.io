import numpy as np
from scipy.linalg import expm

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # Check if f is linear and autonomous
    # 1. Check f(t, 0) == 0
    zero = np.zeros(N)
    f_zero = f(t0, zero)
    if not np.allclose(f_zero, 0, atol=1e-10):
        # Not linear homogeneous, fall back to generic solver
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
        return sol.y[:, -1]

    # 2. Construct A by evaluating f on basis vectors
    # This assumes f(t, y) = A y (autonomous linear)
    A = np.zeros((N, N))
    for j in range(N):
        e = np.zeros(N)
        e[j] = 1.0
        A[:, j] = f(t0, e)
        
    # 3. Verify linearity with y0
    # Check if f(t0, y0) is close to A @ y0
    f_y0 = f(t0, y0)
    A_y0 = A @ y0
    if not np.allclose(f_y0, A_y0, rtol=1e-5, atol=1e-8):
        # Not linear or not autonomous, fall back
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
        return sol.y[:, -1]

    # 4. Solve using matrix exponential or eigen-decomposition
    dt = t1 - t0
    
    # Check symmetry for better performance/accuracy
    if np.allclose(A, A.T, atol=1e-10):
        # Symmetric eigen-decomposition
        w, v = np.linalg.eigh(A)
        # y_final = V * (exp(w*dt) * (V^T * y0))
        v_t_y0 = v.T @ y0
        scaled = np.exp(w * dt) * v_t_y0
        y_final = v @ scaled
    else:
        # General case
        exp_A_dt = expm(A * dt)
        y_final = exp_A_dt @ y0
        
    return y_final
