"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    n = len(y0)
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s kill
    
    # Strategy: Assume the system is a fixed linear system y' = A y + b.
    # The task description explicitly states "fixed 15-dimensional stiff linear system".
    # This allows us to extract A and b using exactly n+1 calls, then solve analytically.
    # This minimizes the call count penalty (log10(calls)) significantly compared to
    # numerical solvers which typically require thousands of calls.
    
    def extract_params(t_val):
        """Extract A and b at time t_val using n+1 calls."""
        zero_vec = np.zeros(n, dtype=float)
        b_val = np.array(f(t_val, zero_vec), dtype=float)
        
        A_val = np.zeros((n, n), dtype=float)
        for i in range(n):
            e_i = np.zeros(n, dtype=float)
            e_i[i] = 1.0
            # f(t, e_i) = A e_i + b = col_i(A) + b
            # col_i(A) = f(t, e_i) - b
            col_i = np.array(f(t_val, e_i), dtype=float) - b_val
            A_val[:, i] = col_i
        return A_val, b_val

    # Sample parameters at t0
    A0, b0 = extract_params(t0)
    
    # Check if we are running out of time (unlikely for 16 calls, but safe)
    if time.time() > deadline:
        return _solve_implicit(f, t_span, y0)

    # Solve constant linear system y' = A0 y + b0 analytically
    # We use the augmented matrix method to handle the non-homogeneous term b.
    # Let z = [y; 1]. Then z' = [A b; 0 0] z.
    # z(t) = exp(t * A_aug) z(0).
    
    A_aug = np.zeros((n + 1, n + 1), dtype=float)
    A_aug[:n, :n] = A0
    A_aug[:n, n] = b0
    # Last row is already zeros
    
    z0 = np.zeros(n + 1, dtype=float)
    z0[:n] = y0
    z0[n] = 1.0
    
    h = t1 - t0
    
    # Compute matrix exponential.
    # scipy.linalg.expm is robust. For n=15, this is fast.
    try:
        M = scipy.linalg.expm(h * A_aug)
        z_end = M @ z0
        y_end = z_end[:n]
        
        # Safety check for finite values
        if np.all(np.isfinite(y_end)):
            return y_end
    except Exception:
        pass
    
    # Fallback if analytic solution fails or produces non-finite results
    return _solve_implicit(f, t_span, y0)

def _solve_implicit(f, t_span, y0):
    """Fallback solver for non-constant or uncertain cases."""
    t0, t1 = t_span
    # Use Radau or BDF for stiffness
    # We don't have a Jacobian here easily unless we re-extract, but that costs calls.
    # However, this path is only taken if the assumption of constant linear system fails,
    # which should be rare given the problem statement.
    
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            max_step=(t1 - t0) / 10.0,
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            max_step=(t1 - t0) / 10.0,
            rtol=1e-10, 
            atol=1e-12
        )
        return sol.y[:, -1]
    except Exception:
        # Last resort: simple Euler (will be inaccurate but returns something)
        y = y0.copy()
        h = (t1 - t0) / 100.0
        t = t0
        for _ in range(100):
            dy = f(t, y)
            y = y + h * dy
            t += h
        return y
# EVOLVE-BLOCK-END
