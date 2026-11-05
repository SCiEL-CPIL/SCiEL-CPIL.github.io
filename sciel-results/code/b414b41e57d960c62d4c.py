import numpy as np
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin before 300s limit
    
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    t0, t1 = t_span
    dt = t1 - t0
    
    # Wrapper to count calls and check deadline
    call_count = 0
    def f_counted(t, y):
        nonlocal call_count
        call_count += 1
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")
        return f(t, y)

    # Strategy: Exploit linearity of the ODE.
    # The convection-diffusion problem is linear and time-invariant:
    # y' = A @ y
    # We can determine A using n function calls (one per column of A) plus
    # a few verification calls.
    # Then solve y(t_end) = expm(A * dt) @ y0 using eigenvalue decomposition.
    
    # Step 1: Determine if the system is linear and time-invariant.
    # First, check linearity by probing with a few vectors.
    # For a linear system f(t, y) = A @ y, we have:
    # f(t, 0) = 0 (homogeneous)
    # f(t, y1 + y2) = f(t, y1) + f(t, y2)
    
    # Check time-invariance: f(t0, y) == f(t_mid, y)
    # Check linearity: f(t, y1 + y2) == f(t, y1) + f(t, y2)
    
    # Use a simple test vector
    test_vec = np.zeros(n)
    test_vec[0] = 1.0
    test_vec2 = np.zeros(n)
    test_vec2[1] = 1.0
    
    # Probe at t0
    f_t0_test1 = f_counted(t0, test_vec)
    f_t0_test2 = f_counted(t0, test_vec2)
    f_t0_sum = f_counted(t0, test_vec + test_vec2)
    
    # Check linearity at t0
    linear_check = np.allclose(f_t0_sum, f_t0_test1 + f_t0_test2, rtol=1e-10, atol=1e-12)
    
    # Check time-invariance
    t_mid = (t0 + t1) / 2.0
    f_tmid_test1 = f_counted(t_mid, test_vec)
    time_inv_check = np.allclose(f_t0_test1, f_tmid_test1, rtol=1e-10, atol=1e-12)
    
    if linear_check and time_inv_check:
        # System is linear and time-invariant: y' = A @ y
        # Determine A by probing each column.
        # A[:, j] = f(t0, e_j) where e_j is the j-th standard basis vector.
        A = np.zeros((n, n))
        for j in range(n):
            ej = np.zeros(n)
            ej[j] = 1.0
            A[:, j] = f_counted(t0, ej)
        
        # Verify with a random vector
        rng = np.random.default_rng(42)
        v_rand = rng.standard_normal(n)
        A_v_rand = A @ v_rand
        f_v_rand = f_counted(t0, v_rand)
        verify_ok = np.allclose(A_v_rand, f_v_rand, rtol=1e-10, atol=1e-12)
        
        if verify_ok:
            # Solve using eigenvalue decomposition
            # y(t_end) = exp(A * dt) @ y0
            try:
                eigenvalues, eigenvectors = np.linalg.eig(A)
                # exp(A*dt) = V @ diag(exp(eigenvalues * dt)) @ V^{-1}
                exp_eigs = np.exp(eigenvalues * dt)
                # Compute V^{-1} @ y0 first for efficiency
                y0_in_eig_basis = np.linalg.solve(eigenvectors, y0)
                y_end = eigenvectors @ (exp_eigs * y0_in_eig_basis)
                
                # Check for numerical issues
                if np.all(np.isfinite(y_end)):
                    return y_end
            except Exception:
                pass
    
    # Fallback: Use a direct numerical integrator if linear assumption fails
    from scipy.integrate import solve_ivp
    
    # Reset call count for fallback (but keep total count for scoring)
    try:
        sol = solve_ivp(
            f_counted, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13,
            max_step=dt/50
        )
        if sol.success:
            return sol.y[:, -1]
    except (TimeoutError, Exception) as e:
        pass

    try:
        sol = solve_ivp(
            f_counted, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12,
            max_step=dt/50
        )
        if sol.success:
            return sol.y[:, -1]
    except (TimeoutError, Exception) as e:
        pass

    try:
        sol = solve_ivp(
            f_counted, 
            t_span, 
            y0, 
            method='BDF', 
            rtol=1e-10, 
            atol=1e-12,
            max_step=dt/50
        )
        if sol.success:
            return sol.y[:, -1]
    except (TimeoutError, Exception) as e:
        pass

    try:
        sol = solve_ivp(
            f_counted, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-10, 
            atol=1e-12,
            max_step=dt/100
        )
        if sol.success:
            return sol.y[:, -1]
    except (TimeoutError, Exception) as e:
        pass

    # If all fail, return y0 to avoid crashing (though score will be 0)
    return y0
