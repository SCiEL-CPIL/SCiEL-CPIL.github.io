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

    # Assume constant coefficients as per task description "fixed ... linear system"
    # Use 1 call for b and n calls for A columns. Total n+1 calls.
    A0, b0 = extract_params(t0)
    
    # Solve constant linear system y' = A0 y + b0 analytically
    # Augment system to handle non-invertible A:
    # z' = [A b; 0 0] z, z(0) = [y0; 1]
    A_aug = np.zeros((n + 1, n + 1), dtype=float)
    A_aug[:n, :n] = A0
    A_aug[:n, n] = b0
    
    z0 = np.zeros(n + 1, dtype=float)
    z0[:n] = y0
    z0[n] = 1.0
    
    h = t1 - t0
    
    # Compute matrix exponential of h * A_aug
    # Using scipy.linalg.expm is robust.
    M = scipy.linalg.expm(h * A_aug)
    
    z_end = M @ z0
    
    y_end = z_end[:n]
    
    if not np.all(np.isfinite(y_end)):
        # Fallback to numerical solver if analytic fails (should not happen)
        return _solve_implicit(f, t_span, y0)
        
    return y_end

def _solve_implicit(f, t_span, y0):
    """Fallback solver for non-constant or uncertain cases."""
    t0, t1 = t_span
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
    else:
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
# EVOLVE-BLOCK-END
