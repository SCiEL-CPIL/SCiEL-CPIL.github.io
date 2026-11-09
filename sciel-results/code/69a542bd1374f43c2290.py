import numpy as np
import scipy.linalg as sla

def integrate(f, t_span, y0):
    t0, t1 = t_span
    T = t1 - t0
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # Get b at t=0
    y_test_0 = np.zeros(N)
    b_0 = f(t0, y_test_0)
    
    # Check if b is constant by checking at t_mid
    t_mid = 0.5 * (t0 + t1)
    b_mid = f(t_mid, y_test_0)
    
    is_affine = np.linalg.norm(b_0) > 1e-10
    is_time_independent = np.linalg.norm(b_0 - b_mid) < 1e-8 * (1 + np.linalg.norm(b_0))
    
    # Extract A columns
    # A[:, i] = f(t0, e_i) - b_0
    A = np.empty((N, N), dtype=float)
    for i in range(N):
        e_i = np.zeros(N)
        e_i[i] = 1.0
        v_i = f(t0, e_i)
        A[:, i] = v_i - b_0
        
    # Verify linearity at t0
    # Use a deterministic vector to avoid random seed issues
    y_rand = np.ones(N)
    # Add some variation to catch more errors
    y_rand = y_rand * np.arange(1, N + 1, dtype=float)
    f_rand = f(t0, y_rand)
    pred_rand = A @ y_rand + b_0
    linearity_ok = np.linalg.norm(f_rand - pred_rand) < 1e-6 * (1 + np.linalg.norm(f_rand))
    
    if is_time_independent and linearity_ok:
        # Solve exactly using eigendecomposition
        if is_affine:
            # Augmented system: z' = M z, z = [y; 1]
            # M = [[A, b_0], [0, 0]]
            M = np.zeros((N + 1, N + 1), dtype=float)
            M[:N, :N] = A
            M[:N, N] = b_0
            z0 = np.zeros(N + 1)
            z0[:N] = y0
            z0[N] = 1.0
            n_dim = N + 1
        else:
            M = A
            z0 = y0
            n_dim = N
            
        # Eigendecomposition
        try:
            vals, vecs = np.linalg.eig(M)
            # Compute exp(lambda * T)
            exp_vals = np.exp(vals * T)
            
            # z_end = V * diag(exp_vals) * V^-1 * z0
            # Let w = V^-1 * z0
            # w = solve(V, z0)
            w = np.linalg.solve(vecs, z0)
            # w_scaled = exp_vals * w
            w_scaled = exp_vals * w
            # z_end = V * w_scaled
            z_end = vecs @ w_scaled
            
            # Ensure real output
            z_end = np.real_if_close(z_end, tol=1000)
            if np.iscomplexobj(z_end):
                # If still complex, take real part (imaginary should be numerical noise)
                z_end = z_end.real
                
            if is_affine:
                return z_end[:N]
            else:
                return z_end
                
        except np.linalg.LinAlgError:
            # Fallback if eig fails (rare)
            pass
    else:
        # Fallback to robust adaptive solver
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12, max_step=T/100)
        if sol.success:
            return sol.y[:, -1]
        else:
            # Last resort: RK4 with many steps
            steps = 10000
            h = T / steps
            t, y = t0, y0.copy()
            for _ in range(steps):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h/2 * k3) # Wait, standard RK4 is y + h*k3? No, y + h*k3 is for t+h?
                # k4 = f(t+h, y + h*k3)
                # y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                # My previous code had k4 = f(t+h, y+h*k3).
                # Let's keep the fallback simple and correct.
                k4 = f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
            return y
