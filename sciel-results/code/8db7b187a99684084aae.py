import numpy as np
import scipy.linalg as sla
import scipy.integrate as integrate_scipy
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    T = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 285.0

    # 1. Attempt Exact Solution for Linear/Affine Time-Invariant Systems
    A = None
    c = None
    is_valid_linear = False
    
    try:
        # Probe at t0
        y_zero = np.zeros(n)
        c0 = f(t0, y_zero)
        
        if not np.all(np.isfinite(c0)):
            raise ValueError("Non-finite f")
            
        c = c0.copy()
        A = np.zeros((n, n))
        
        # Extract columns of A
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            val = f(t0, e_j)
            if not np.all(np.isfinite(val)):
                raise ValueError("Non-finite f")
            A[:, j] = val - c
            
            if time.time() > deadline:
                raise TimeoutError()

        # Verify Linearity at t0
        # Use a random vector
        rng = np.random.default_rng(123)
        y_test = rng.standard_normal(n)
        f_test_t0 = f(t0, y_test)
        pred_t0 = A @ y_test + c
        err_t0 = np.linalg.norm(f_test_t0 - pred_t0)
        scale_t0 = np.linalg.norm(A) * np.linalg.norm(y_test) + np.linalg.norm(c) + 1e-12
        if err_t0 > 1e-9 * scale_t0:
            raise ValueError("Non-linear at t0")
            
        # Verify Time-Invariance at t1 (or midpoint if T is small)
        t_check = t1 if T > 1e-6 else t0 + 1e-6
        f_test_t1 = f(t_check, y_test)
        pred_t1 = A @ y_test + c
        err_t1 = np.linalg.norm(f_test_t1 - pred_t1)
        scale_t1 = scale_t0
        if err_t1 > 1e-9 * scale_t1:
            raise ValueError("Time-dependent")
            
        is_valid_linear = True

    except Exception:
        is_valid_linear = False
        A = None
        c = None

    if is_valid_linear:
        # Construct augmented matrix for affine system y' = Ay + c
        # State vector z = [y; 1]
        # z' = [A c; 0 0] z
        n_aug = n + 1
        A_aug = np.zeros((n_aug, n_aug))
        A_aug[:n, :n] = A
        A_aug[:n, n] = c
        
        y0_aug = np.zeros(n_aug)
        y0_aug[:n] = y0
        y0_aug[n] = 1.0
        
        try:
            # Use eigendecomposition for potentially better stability with stiff systems
            # Compute eigenvalues and eigenvectors
            eigvals, eigvecs = sla.eig(A_aug)
            
            # Check if eigenvalues are well-conditioned
            # If not, fall back to expm
            min_re = np.min(np.real(eigvals))
            max_re = np.max(np.real(eigvals))
            
            # If the system is stable (all real parts <= 0) or not too stiff, try eig
            # For convection-diffusion, eigenvalues are typically real and negative
            if np.all(np.imag(eigvals) < 1e-10) or np.all(np.abs(eigvals) < 1e6):
                exp_eigvals = np.exp(eigvals * T)
                
                # Compute V * diag(exp(lambda_i * T)) * V^{-1}
                # This is equivalent to expm(A_aug * T)
                # But we can compute it as:
                # y_final = V * (exp(eigvals * T) * (V^{-1} * y0_aug))
                
                try:
                    inv_V = sla.inv(eigvecs)
                    y0_transformed = inv_V @ y0_aug
                    y_final_transformed = exp_eigvals * y0_transformed
                    y_final_aug = eigvecs @ y_final_transformed
                    y_final = y_final_aug[:n]
                    
                    if np.all(np.isfinite(y_final)):
                        return y_final
                except Exception:
                    pass
            
            # Fallback to expm if eig fails or is unstable
            exp_aug = sla.expm(A_aug * T)
            y_final_aug = exp_aug @ y0_aug
            y_final = y_final_aug[:n]
            
            if np.all(np.isfinite(y_final)):
                return y_final
        except Exception:
            pass # Fall back

    # 2. Fallback: Numerical Integration
    def ode_func(t, y):
        return f(t, y)

    # Try LSODA first (stiff/non-stiff adaptive)
    try:
        sol = integrate_scipy.solve_ivp(
            ode_func, 
            t_span, 
            y0, 
            method='LSODA', 
            rtol=1e-10, 
            atol=1e-12,
            max_step=T/100.0
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass

    # Try Radau (implicit, stiff)
    try:
        sol = integrate_scipy.solve_ivp(
            ode_func, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass

    # Last resort: RK45 with tight tolerance
    try:
        sol = integrate_scipy.solve_ivp(
            ode_func, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-12, 
            atol=1e-14,
            max_step=T/1000.0
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass

    return y0
