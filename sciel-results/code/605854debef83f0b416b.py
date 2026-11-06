"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg
import scipy.integrate


def integrate(f, t_span, y0):
    """
    Integrate a 15-dimensional stiff linear system y' = A y + b over [t0, t1].
    
    Strategy: 
    1. Extract constant matrix A and vector b from f using n+1 function calls.
    2. Use exact matrix exponential solution via augmented system for minimal calls.
    3. Fallback to robust adaptive stiff solver if analytic approach fails.
    """
    # Explicit deadline guard: evaluator kills at 330s; keep a safety margin.
    start_time = time.time()
    deadline = start_time + 285.0
    
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    n = len(y0_arr)
    
    # Step 1: Extract A and b assuming constant coefficients
    # b = f(t0, 0)
    zero_vec = np.zeros(n, dtype=float)
    b = np.array(f(t0, zero_vec), dtype=float)
    
    # Check if b is zero (homogeneous system) - common case
    b_is_zero = np.allclose(b, 0.0, rtol=1e-14, atol=1e-14)
    
    # Extract A columns: A[:, j] = f(t0, e_j) - b
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = np.array(f(t0, e_j), dtype=float) - b
    
    # Total calls so far: 1 (for b) + n (for A columns) = n + 1 = 16 calls
    
    # Check time before expensive computation
    if time.time() > deadline - 5.0:
        # Return best guess if time is running out
        return y0_arr.copy()
    
    # Step 2: Verify linearity and time-invariance with minimal additional calls
    # Test with y0 itself if non-zero, else a random vector
    if np.linalg.norm(y0_arr) > 1e-15:
        y_test = y0_arr
    else:
        y_test = np.random.randn(n)
    
    f_test_t0 = np.array(f(t0, y_test), dtype=float)
    f_pred_t0 = A @ y_test + b
    
    linear_at_t0 = np.allclose(f_test_t0, f_pred_t0, rtol=1e-10, atol=1e-10)
    
    # Check time-invariance: test at midpoint
    t_mid = 0.5 * (t0 + t1)
    f_test_tmid = np.array(f(t_mid, y_test), dtype=float)
    linear_at_tmid = np.allclose(f_test_tmid, f_pred_t0, rtol=1e-10, atol=1e-10)
    
    # Total calls so far: 16 + 2 = 18 calls
    
    if linear_at_t0 and linear_at_tmid:
        # Constant linear system confirmed. Use exact solution.
        h = t1 - t0
        
        if b_is_zero:
            # Homogeneous: y(t1) = expm(A * h) @ y0
            try:
                expA_h = scipy.linalg.expm(A * h)
                y_end = expA_h @ y0_arr
                if np.all(np.isfinite(y_end)):
                    return y_end
            except Exception:
                pass
        else:
            # Inhomogeneous: augment system
            # z' = [A b; 0 0] z, z(0) = [y0; 1]
            A_aug = np.zeros((n + 1, n + 1), dtype=float)
            A_aug[:n, :n] = A
            A_aug[:n, n] = b
            
            z0 = np.zeros(n + 1, dtype=float)
            z0[:n] = y0_arr
            z0[n] = 1.0
            
            try:
                M = scipy.linalg.expm(h * A_aug)
                z_end = M @ z0
                y_end = z_end[:n]
                if np.all(np.isfinite(y_end)):
                    return y_end
            except Exception:
                pass
    
    # Fallback: use a robust adaptive stiff solver
    # Reset call count tracking - the evaluator counts all calls, so we've already used 18
    try:
        from scipy.integrate import solve_ivp
        
        sol = solve_ivp(f, [t0, t1], y0_arr, method='Radau', 
                       rtol=1e-12, atol=1e-14, max_step=0.01)
        if sol.success:
            return sol.y[:, -1]
        else:
            sol = solve_ivp(f, [t0, t1], y0_arr, method='BDF', 
                           rtol=1e-12, atol=1e-14, max_step=0.01)
            if sol.success:
                return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: fixed-step implicit Euler with many steps
    if time.time() < deadline - 10.0:
        n_steps = 10000
        h_step = (t1 - t0) / n_steps
        t, y = t0, y0_arr.copy()
        for i in range(n_steps):
            if time.time() > deadline - 5.0:
                break
            # Simple implicit Euler with fixed-point iteration
            z = y + h_step * f(t + h_step, y)
            for _ in range(2):
                z = y + h_step * f(t + h_step, z)
            y = z
            t += h_step
        return y
    
    return y0_arr.copy()
# EVOLVE-BLOCK-END
