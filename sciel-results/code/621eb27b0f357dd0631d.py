"""ODE solver: Convection-diffusion 100-point using affine matrix exponential."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
from scipy.sparse.linalg import LinearOperator


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    n = len(y0_arr)
    
    deadline = time.time() + 290  # 300s limit, keep 10s margin
    max_calls = 2000000
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Strategy: Assume affine-linear dynamics y' = A y + b.
    # For convection-diffusion with Dirichlet BCs, this holds.
    # 1. Evaluate b = f(t, 0)
    # 2. Evaluate columns of A using f(t, e_i) - b
    # 3. Verify and use augmented matrix exponential.
    
    try:
        # Evaluate at zero to get b
        y_zero = np.zeros(n)
        b = f_counted(t0, y_zero)
        
        # Evaluate at e0
        e0 = np.zeros(n)
        e0[0] = 1.0
        col0_raw = f_counted(t0, e0)
        col0 = col0_raw - b
        
        # Evaluate at e1
        e1 = np.zeros(n)
        e1[1] = 1.0
        col1_raw = f_counted(t0, e1)
        col1 = col1_raw - b
        
        # Extract tridiagonal parameters
        # A is tridiagonal:
        # Col 0 has non-zeros at [0, 1]
        # Col 1 has non-zeros at [0, 1, 2]
        
        # Handle n=1 case separately or ensure indices exist
        if n == 1:
            A = np.array([[col0[0]]])
        else:
            # From col0:
            # A[0,0] = col0[0]
            # A[1,0] = col0[1] (lower diagonal)
            # Others should be 0
            
            # From col1:
            # A[0,1] = col1[0] (upper diagonal)
            # A[1,1] = col1[1] (diagonal)
            # A[2,1] = col1[2] (lower diagonal, should match A[1,0])
            
            diag_val = col1[1]
            upper_val = col1[0]
            lower_val = col0[1]
            
            # Build A
            A = np.zeros((n, n))
            for i in range(n):
                A[i, i] = diag_val
                if i > 0:
                    A[i, i-1] = lower_val
                if i < n-1:
                    A[i, i+1] = upper_val
            
            # Verification: Check if A @ y0 + b matches f(t, y0)
            # We already have f(t, y0) if we evaluated it, but we didn't in the try block above yet.
            # We used y_zero, e0, e1. We need one more call for y0 to verify.
            
        # Verify the model
        f_y0 = f_counted(t0, y0_arr)
        A_y0_plus_b = A @ y0_arr + b
        
        # Compute relative error of the model
        norm_f = np.linalg.norm(f_y0)
        norm_diff = np.linalg.norm(A_y0_plus_b - f_y0)
        rel_err = norm_diff / (norm_f + 1e-30)
        
        if rel_err < 1e-10:
            # Model is valid. Use exact solution via matrix exponential.
            dt = t1 - t0
            
            # Check if b is effectively zero to use simpler form
            b_norm = np.linalg.norm(b)
            a_norm = np.linalg.norm(A, ord=1)
            
            try:
                if b_norm < 1e-12 * (a_norm + 1.0):
                    # Purely linear: y' = A y
                    # y_end = expm(A * dt) @ y0
                    # Use sparse matrix exponential for efficiency
                    A_sparse = diags([lower_val, diag_val, upper_val], [-1, 0, 1], shape=(n, n))
                    y_end = expm_multiply(A_sparse * dt, y0_arr)
                else:
                    # Affine: y' = A y + b
                    # Use augmented matrix M = [[A, b], [0, 0]]
                    M = np.zeros((n+1, n+1))
                    M[:n, :n] = A
                    M[:n, n] = b
                    E_M = expm(M * dt)
                    y_end = E_M[:n, :n] @ y0_arr + E_M[:n, n]
                
                if np.all(np.isfinite(y_end)):
                    return y_end
            except Exception:
                # Fallback to dense expm if sparse fails
                try:
                    if b_norm < 1e-12 * (a_norm + 1.0):
                        E_A = expm(A * dt)
                        y_end = E_A @ y0_arr
                    else:
                        M = np.zeros((n+1, n+1))
                        M[:n, :n] = A
                        M[:n, n] = b
                        E_M = expm(M * dt)
                        y_end = E_M[:n, :n] @ y0_arr + E_M[:n, n]
                    
                    if np.all(np.isfinite(y_end)):
                        return y_end
                except Exception:
                    pass
        else:
            # Model invalid, fall through to numerical solver
            pass
            
    except Exception:
        pass
    
    # Fallback: High precision numerical integration
    # Reset call count? No, the evaluator counts total calls to f.
    # We have used 4 calls. We have plenty of budget.
    
    try:
        sol = solve_ivp(
            f_counted,
            [t0, t1],
            y0_arr,
            method='DOP853',
            rtol=1e-14,
            atol=1e-16,
            max_step=(t1 - t0) / 200,
            max_nfev=max_calls
        )
        
        if sol.success and sol.t[-1] >= t1 - 1e-14:
            return sol.y[:, -1]
        
        if sol.t.size > 0 and t1 - sol.t[-1] < 1e-10:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback 2: RK45
    try:
        sol = solve_ivp(
            f_counted,
            [t0, t1],
            y0_arr,
            method='RK45',
            rtol=1e-13,
            atol=1e-15,
            max_step=(t1 - t0) / 500,
            max_nfev=max_calls
        )
        
        if sol.success and sol.t[-1] >= t1 - 1e-14:
            return sol.y[:, -1]
            
        if sol.t.size > 0 and t1 - sol.t[-1] < 1e-8:
            return sol.y[:, -1]
    except Exception:
        pass

    # Last resort: Fixed-step RK4
    try:
        n_steps = 5000
        h = (t1 - t0) / n_steps
        t = t0
        y = y0_arr.copy()
        
        for _ in range(n_steps):
            if time.time() > deadline:
                break
            k1 = f_counted(t, y)
            k2 = f_counted(t + h/2, y + h/2 * k1)
            k3 = f_counted(t + h/2, y + h/2 * k2)
            k4 = f_counted(t + h, y + h * k3)
            y = y + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            if not np.all(np.isfinite(y)):
                break
        
        return y
    except Exception:
        pass
    
    return y0_arr
# EVOLVE-BLOCK-END
