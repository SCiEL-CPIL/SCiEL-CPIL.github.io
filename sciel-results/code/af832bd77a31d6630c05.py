import numpy as np
import scipy.linalg
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    timeout = 290.0

    # Strategy: Try to exploit tridiagonal/Toeplitz structure to minimize f calls.
    # 1. Extract boundary columns and one interior column.
    # 2. Assume interior is Toeplitz.
    # 3. Reconstruct matrix.
    # 4. Verify with a test vector.
    # 5. If verified, use expm.
    # 6. If not, fallback to full extraction or solver.

    if n == 0:
        return y0

    # Indices
    i_mid = n // 2
    
    # Extract columns
    e_0 = np.zeros(n); e_0[0] = 1.0
    e_end = np.zeros(n); e_end[-1] = 1.0
    e_mid = np.zeros(n); e_mid[i_mid] = 1.0
    
    col_0 = f(0.0, e_0)
    col_end = f(0.0, e_end)
    col_mid = f(0.0, e_mid)
    
    # Check timeout
    if time.time() - start_time > timeout:
        sol = scipy.integrate.solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-12, atol=1e-14)
        return sol.y[:, -1]

    # Reconstruct A assuming tridiagonal with constant interior
    A_guess = np.zeros((n, n))
    
    # Diagonals
    # Interior diagonal from mid
    d_int = col_mid[i_mid]
    # Boundary diagonals from 0 and end
    d_0 = col_0[0]
    d_end = col_end[-1]
    
    A_guess[0, 0] = d_0
    A_guess[-1, -1] = d_end
    if n > 2:
        A_guess[1:-1, 1:-1] = np.eye(n-2) * d_int
    
    # Super-diagonals (A[i, i+1])
    # From col_mid, the element at row i_mid-1, col i_mid is A[i_mid-1, i_mid]
    if n > 1:
        u_int = col_mid[i_mid - 1] if i_mid > 0 else 0.0
        # Fill super diagonal
        # A[i, i+1] for i=0 to n-2
        # Assume constant u_int for all i
        for i in range(n - 1):
            A_guess[i, i + 1] = u_int
            
    # Sub-diagonals (A[i+1, i])
    # From col_mid, the element at row i_mid+1, col i_mid is A[i_mid+1, i_mid]
    if n > 1:
        l_int = col_mid[i_mid + 1] if i_mid < n - 1 else 0.0
        # Fill sub diagonal
        # A[i+1, i] for i=0 to n-2
        for i in range(n - 1):
            A_guess[i + 1, i] = l_int

    # Verify linearity and structure
    y_test = np.ones(n)
    y_expected = A_guess @ y_test
    y_actual = f(0.0, y_test)
    
    diff = np.linalg.norm(y_actual - y_expected)
    scale = np.linalg.norm(y_expected)
    
    # Tolerance for verification
    tol = 1e-10 * (1.0 + scale)
    
    if diff < tol:
        # Solve using matrix exponential
        M = A_guess * dt
        try:
            expM = scipy.linalg.expm(M)
            y_end = expM @ y0
            if np.all(np.isfinite(y_end)):
                return y_end
        except Exception:
            pass # Fall through to solver if expm fails
            
    # Fallback: Full extraction if structure assumption failed
    # Extract full A
    A = np.zeros((n, n))
    for j in range(n):
        if time.time() - start_time > timeout:
            break
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(0.0, e_j)
        
    if time.time() - start_time > timeout:
        sol = scipy.integrate.solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-12, atol=1e-14)
        return sol.y[:, -1]

    # Verify full A
    y_test2 = np.random.rand(n)
    y_expected2 = A @ y_test2
    y_actual2 = f(0.0, y_test2)
    diff2 = np.linalg.norm(y_actual2 - y_expected2)
    scale2 = np.linalg.norm(y_expected2)
    
    if diff2 < 1e-10 * (1.0 + scale2):
        M = A * dt
        try:
            expM = scipy.linalg.expm(M)
            y_end = expM @ y0
            if np.all(np.isfinite(y_end)):
                return y_end
        except Exception:
            pass

    # Final Fallback: High accuracy adaptive solver
    sol = scipy.integrate.solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-12, atol=1e-14)
    return sol.y[:, -1]
