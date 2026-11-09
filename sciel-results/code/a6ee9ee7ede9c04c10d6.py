import numpy as np
from scipy.linalg import eig, inv
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    start_time = time.time()
    deadline = start_time + 285.0

    # Extract A and b
    y_zero = np.zeros(n)
    b = f(t0, y_zero)
    
    A = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        A[:, j] = f(t0, ej) - b
        
    # Verify LTI assumption
    # Test with a random vector and a different time
    t_test = (t0 + t1) / 2
    y_test = np.random.randn(n)
    f_test = f(t_test, y_test)
    f_linear = A @ y_test + b
    
    is_lti = np.allclose(f_test, f_linear, rtol=1e-8, atol=1e-10)
    
    if is_lti:
        # Use eigen-decomposition for high precision and speed
        # Handle potential non-diagonalizability or complex eigenvalues
        try:
            eigvals, eigvecs = eig(A)
            # Check condition number of eigvecs
            cond = np.linalg.cond(eigvecs)
            if cond < 1e12: # Well-conditioned
                eigvecs_inv = inv(eigvecs)
                
                dt = t1 - t0
                # e^(A dt) y0
                # e^(D dt) is diagonal with exp(eigvals * dt)
                exp_vals = np.exp(eigvals * dt)
                y_homo = eigvecs @ (exp_vals * (eigvecs_inv @ y))
                
                # Integral term: int_0^dt e^(A tau) b dtau
                # = V * diag( (e^(d_i dt) - 1) / d_i ) * V^-1 * b
                # Handle d_i == 0 case
                integral_diag = np.zeros_like(eigvals, dtype=complex)
                mask = np.abs(eigvals) > 1e-12
                integral_diag[mask] = (exp_vals[mask] - 1) / eigvals[mask]
                integral_diag[~mask] = 1.0 # limit as d->0 is 1 (integral of 1 is dt? No. e^0=1. (1-1)/0 is 0/0. Limit is 1? No. 
                # Let's re-evaluate: integral_0^t e^(d tau) dtau = (e^(dt)-1)/d. If d=0, e^0=1, integral is t.
                # So if d_i is close to 0, term is dt.
                integral_diag[~mask] = dt
                
                b_proj = eigvecs_inv @ b
                y_particular = eigvecs @ (integral_diag * b_proj)
                
                y_final = y_homo + y_particular
                y_final = np.real_if_close(y_final, tol=1000)
                
                if time.time() < deadline:
                    return y_final
            else:
                # If ill-conditioned, fallback to expm or solve_ivp
                pass
        except Exception:
            pass
        
        # Fallback within LTI branch if eigen fails or is bad: use expm_multiply
        # y(t) = expm(A dt) @ y + ...
        # But we need the integral. 
        # Let's just use solve_ivp if eigen is problematic, as it's robust.
        
    # Fallback: solve_ivp
    # Use LSODA which switches between stiff/non-stiff
    sol = solve_ivp(f, t_span, y, method='LSODA', rtol=1e-12, atol=1e-14)
    return sol.y[:, -1]
