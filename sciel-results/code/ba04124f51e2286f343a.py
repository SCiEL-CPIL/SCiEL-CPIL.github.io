"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla
import scipy.integrate as integrate
import time

def integrate(f, t_span, y0):
    """
    Solves the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses a hybrid approach:
    1. Attempts to identify if the system is linear and time-independent (f(t, y) = A y).
       If so, uses matrix exponential for machine-precision accuracy with minimal f-calls.
    2. Falls back to a high-precision implicit solver (Radau) for general/stiff cases.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: 300s limit, keep margin
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Strategy 1: Detect linear time-invariant system
    # Check if f(t, 0) == 0
    f0 = f(t0, np.zeros(n))
    is_homogeneous = np.allclose(f0, 0.0, atol=1e-12)
    
    if is_homogeneous:
        # Reconstruct A by probing basis vectors
        # We need n calls to get columns of A
        A = np.zeros((n, n), dtype=float)
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            A[:, j] = f(t0, e_j)
        
        # Verify time-invariance and linearity with a random test vector
        # Pick a mid-point in time
        t_mid = (t0 + t1) / 2.0
        # Random test vector
        rng = np.random.default_rng(42)
        y_test = rng.normal(size=n)
        
        f_test_direct = f(t_mid, y_test)
        f_test_linear = A @ y_test
        
        # Also check at t1
        f_test_end = f(t1, y_test)
        
        if np.allclose(f_test_direct, f_test_linear, rtol=1e-10, atol=1e-12) and \
           np.allclose(f_test_end, f_test_linear, rtol=1e-10, atol=1e-12):
            
            # System is linear and time-independent: y' = A y
            # Solution: y(t1) = exp(A * (t1 - t0)) @ y0
            dt = t1 - t0
            # Use scipy.linalg.expm for dense matrix (n=100 is small enough)
            # expm_multiply is also an option but expm is very fast for 100x100
            expA = sla.expm(A * dt)
            y_end = expA @ y0
            
            # Sanity check: ensure no NaNs/Infs
            if np.all(np.isfinite(y_end)):
                return y_end
            else:
                # Fallback if expm failed (unlikely)
                pass
        else:
            # Not perfectly linear or time-dependent, fall through to general solver
            pass
    else:
        # Not homogeneous, likely non-linear or time-dependent forcing
        pass

    # Strategy 2: General high-precision integration
    # Use Radau for stiff problems (convection-diffusion is often stiff)
    # Tight tolerances to maximize accuracy
    # Max steps to prevent infinite loops, but Radau is adaptive
    
    # We have already used ~n+1 calls in the detection phase if it was homogeneous-ish
    # If we are here, we just use the solver. The solver will count its own calls.
    
    try:
        sol = integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-12, 
            max_step=(t1-t0)/100.0 # Limit step size to ensure accuracy if needed, though adaptive handles it
        )
        
        if sol.success:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass
    
    # Strategy 3: Fallback to RK45 if Radau fails (less likely for stiff, but good backup)
    try:
        sol = integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-12, 
            atol=1e-12
        )
        if sol.success:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    # Strategy 4: Last resort, manual high-step RK4 (similar to seed but more steps if time permits)
    # This is unlikely to be reached if SciPy is working.
    steps = 5000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        y = y + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
            
    return y
# EVOLVE-BLOCK-END
