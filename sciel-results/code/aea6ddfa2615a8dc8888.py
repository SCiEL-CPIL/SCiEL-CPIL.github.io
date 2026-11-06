import numpy as np
import scipy.linalg
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    n = len(y0)
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 295.0 # Safety margin

    def extract_params(t_val):
        """Extract A and b at time t_val using n+1 calls."""
        zero_vec = np.zeros(n, dtype=float)
        b_val = np.array(f(t_val, zero_vec), dtype=float)
        
        A_val = np.zeros((n, n), dtype=float)
        for i in range(n):
            e_i = np.zeros(n, dtype=float)
            e_i[i] = 1.0
            col_i = np.array(f(t_val, e_i), dtype=float) - b_val
            A_val[:, i] = col_i
        return A_val, b_val

    # Check if the system is constant coefficient
    # This costs 2*(n+1) calls
    A0, b0 = extract_params(t0)
    A1, b1 = extract_params(t1)
    
    # Check similarity
    # Use a relative tolerance
    tol = 1e-8
    is_constant = (np.linalg.norm(A0 - A1, ord='fro') < tol * (np.linalg.norm(A0, ord='fro') + 1e-12) and
                   np.linalg.norm(b0 - b1) < tol * (np.linalg.norm(b0) + 1e-12))
    
    if is_constant:
        # Use analytic solution
        A_aug = np.zeros((n + 1, n + 1), dtype=float)
        A_aug[:n, :n] = A0
        A_aug[:n, n] = b0
        
        z0 = np.zeros(n + 1, dtype=float)
        z0[:n] = y0
        z0[n] = 1.0
        
        h = t1 - t0
        M = scipy.linalg.expm(h * A_aug)
        z_end = M @ z0
        y_end = z_end[:n]
        
        if np.all(np.isfinite(y_end)):
            return y_end
        # Fallback if analytic fails

    # Numerical solver for time-dependent or failed analytic
    # Use Radau for stiff systems
    sol = scipy.integrate.solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-10, 
        atol=1e-12,
        max_step=(t1 - t0) / 10.0 # Prevents huge steps if the solver is lazy
    )
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback to BDF if Radau fails
        sol2 = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            rtol=1e-10, 
            atol=1e-12,
            max_step=(t1 - t0) / 10.0
        )
        return sol2.y[:, -1]
