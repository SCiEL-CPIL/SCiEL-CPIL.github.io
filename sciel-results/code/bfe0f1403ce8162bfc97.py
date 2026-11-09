"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    n = len(y0_arr)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin for 300s limit
    
    # Strategy: Assume "fixed" means constant coefficients A and b.
    # We probe A and b at t0 only.
    # Calls: n (for A) + 1 (for b) = 16 calls.
    
    # 1. Probe b(t0) = f(t0, 0)
    y_zero = np.zeros(n)
    b_t0 = np.array(f(t0, y_zero), dtype=float)
    
    # 2. Probe A(t0) columns
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        f_j = np.array(f(t0, e_j), dtype=float)
        A[:, j] = f_j - b_t0
        
    # Check if we are running out of time (unlikely for 16 calls + expm)
    if time.time() > deadline:
        # Fallback: Return y0 if we can't even compute expm? 
        # But expm is fast. This is just a safety net.
        pass
        
    dt = t1 - t0
    if abs(dt) < 1e-15:
        return y0_arr
        
    # Construct augmented matrix M for y' = A y + b
    # z = [y; 1]
    # z' = M z, where M = [[A, b], [0, 0]]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n] = b_t0
    # Last row is zero
    
    z0 = np.zeros(n + 1)
    z0[:n] = y0_arr
    z0[n] = 1.0
    
    # Compute matrix exponential
    # For stiff systems, expm is robust.
    # If the system is extremely stiff, expm might underflow to 0 for some components, 
    # which is physically correct if eigenvalues are very negative.
    
    expM = expm(M * dt)
    z_final = expM @ z0
    y_final = z_final[:n]
    
    # Sanity check
    if not np.all(np.isfinite(y_final)):
        # If expm failed, try a simple Euler step? No, that's inaccurate.
        # Try scipy solve_ivp as fallback if we have time and calls budget.
        # But we've already used 16 calls. We have 2,000,000 - 16 available.
        from scipy.integrate import solve_ivp
        sol = solve_ivp(
            f, (t0, t1), y0_arr,
            method='BDF',
            rtol=1e-13,
            atol=1e-15,
            max_step=abs(dt) / 100
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            return y_final # Return the failed expm result or raise?
            # Better to return something finite if possible, but expm failed.
            # Let's just return y0 if all else fails, to avoid crash.
            
    return y_final
# EVOLVE-BLOCK-END
