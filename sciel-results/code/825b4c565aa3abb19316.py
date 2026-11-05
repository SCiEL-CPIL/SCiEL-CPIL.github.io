import numpy as np
from scipy.linalg import expm
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = y0.size
    dt = t1 - t0
    start = time.time()
    deadline = start + 280.0  # 20s safety margin

    if dt == 0:
        return y0.copy()

    # Helper to extract A(t) assuming f(t, y) = A(t) y
    def get_A(t):
        A = np.empty((n, n), dtype=float)
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            A[:, j] = f(t, e)
        return A

    # 1. Check for time-independent linear system
    # Extract A at t0
    A0 = get_A(t0)
    
    # Check homogeneity: f(t, 0) should be 0
    z0 = f(t0, np.zeros(n))
    hom_err = np.linalg.norm(z0)
    
    # Check time-independence: A(t1) and A(mid) should match A0
    A1 = get_A(t1)
    Amid = get_A(0.5 * (t0 + t1))
    
    time_indep_err = max(np.linalg.norm(A0 - A1), np.linalg.norm(A0 - Amid))
    
    # If it looks like a constant linear system
    if hom_err < 1e-12 and time_indep_err < 1e-12:
        M = expm(A0 * dt)
        return M @ y0

    # 2. Check for time-dependent linear system
    # Verify linearity at a few points
    np.random.seed(42)
    y_test1 = np.random.randn(n)
    y_test2 = np.random.randn(n)
    alpha = 2.5
    
    # Test at t_mid
    t_test = 0.5 * (t0 + t1)
    A_test = get_A(t_test)
    
    pred_sum = A_test @ (y_test1 + y_test2)
    actual_sum = f(t_test, y_test1 + y_test2)
    err_sum = np.linalg.norm(pred_sum - actual_sum) / (1e-30 + np.linalg.norm(actual_sum))
    
    pred_alpha = A_test @ (alpha * y_test1)
    actual_alpha = f(t_test, alpha * y_test1)
    err_alpha = np.linalg.norm(pred_alpha - actual_alpha) / (1e-30 + np.linalg.norm(actual_alpha))
    
    # Also check homogeneity at t_test
    z_test = f(t_test, np.zeros(n))
    hom_err_test = np.linalg.norm(z_test)
    
    is_linear = (err_sum < 1e-10 and err_alpha < 1e-10 and hom_err_test < 1e-12)
    
    if is_linear:
        # Use Piecewise Constant Approximation (PCA) with Matrix Exponentials
        # Midpoint rule for A(t) to get 2nd order accuracy in h
        # Error is O(h^2)
        
        # We want to maximize K to minimize error, subject to:
        # 1. Total calls < 2,000,000
        # 2. Time < deadline
        
        # Each step requires n calls to f to build A.
        # Total calls = K * n
        # Max K by calls: 2,000,000 / n
        max_K_by_calls = 2000000 // n
        
        # Estimate time for one step
        # One step: 1 get_A (n calls to f) + 1 expm (15x15)
        # Let's time a single step roughly
        t_start_step = time.time()
        t_probe = t0 + 0.1 * dt
        A_probe = get_A(t_probe)
        M_probe = expm(A_probe * (dt / 1000))
        t_end_step = time.time()
        time_per_step = t_end_step - t_start_step
        
        if time_per_step == 0:
            time_per_step = 1e-6 # Fallback minimum estimate
            
        # Max K by time: (deadline - time.time()) / time_per_step
        time_remaining = deadline - time.time()
        max_K_by_time = int(time_remaining / time_per_step)
        
        # Choose K, capped by both limits, but also leave some buffer
        # We don't want to hit the call limit exactly, so 90% of max calls
        K = min(max_K_by_calls, max_K_by_time)
        
        # Ensure K is at least some minimum to be reasonable, but not so high it times out
        # If K is 0, fallback
        if K < 100:
             K = 100
             
        # Safety: don't exceed 90% of call limit to be safe
        K = min(K, int(0.9 * max_K_by_calls))
        
        h = dt / K
        y = y0.copy()
        
        # Pre-allocate? No, A changes.
        # Loop
        for k in range(K):
            if time.time() > deadline:
                break
                
            t_mid = t0 + (k + 0.5) * h
            A_k = get_A(t_mid)
            M_k = expm(A_k * h)
            y = M_k @ y
            
        return y

    # 3. Fallback: Non-linear or detection failed
    # Use solve_ivp with Radau
    
    sol = solve_ivp(f, (t0, t1), y0, method='Radau', rtol=1e-10, atol=1e-12,
                    max_step=0.01)
    
    if not sol.success:
        sol = solve_ivp(f, (t0, t1), y0, method='BDF', rtol=1e-10, atol=1e-12,
                        max_step=0.01)
        
    return sol.y[:, -1]
