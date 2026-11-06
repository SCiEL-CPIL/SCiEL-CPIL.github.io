import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = 300.0
    
    # Estimate Jacobian using central differences
    eps = 1e-7 * max(1.0, np.max(np.abs(y0)))
    
    f0 = f(t0, y0)
    
    A = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps
        fp = f(t0, y0 + e)
        fm = f(t0, y0 - e)
        A[:, j] = (fp - fm) / (2 * eps)
        
        if time.time() > deadline - 10:
            break

    # Check linearity: f(t, y) == A @ y + b ?
    b = f0 - A @ y0
    
    # Verify with a test point
    np.random.seed(42)
    v_test = np.random.randn(n)
    f_test = f(t0, v_test)
    pred_test = A @ v_test + b
    
    err_test = np.linalg.norm(f_test - pred_test)
    scale = max(1.0, np.linalg.norm(f_test))
    
    is_linear = err_test < 1e-6 * scale
    
    if is_linear:
        dt = t1 - t0
        try:
            # Augmented matrix for non-homogeneous linear system
            # [ A  b ]
            # [ 0  0 ]
            A_aug = np.zeros((n + 1, n + 1))
            A_aug[:n, :n] = A
            A_aug[:n, n] = b
            
            M_aug = expm(A_aug * dt)
            
            # y_end = M_aug[0, :n] @ y0 + M_aug[0, n]
            y_end = M_aug[0, :n] @ y0 + M_aug[0, n]
            
            if np.all(np.isfinite(y_end)):
                return y_end
        except Exception:
            pass
            
    # Fallback to robust implicit solver
    sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
    if sol.success:
        return sol.y[:, -1]
    else:
        return sol.y[:, -1] if sol.y.size > 0 else y0
