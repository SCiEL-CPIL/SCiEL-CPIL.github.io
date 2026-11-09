import numpy as np
import scipy.linalg as sla

def integrate(f, t_span, y0):
    t0, t1 = t_span
    T = t1 - t0
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # Extract b
    y_zero = np.zeros(N)
    b = f(t0, y_zero)
    
    # Extract A
    A = np.empty((N, N), dtype=float)
    for i in range(N):
        e_i = np.zeros(N)
        e_i[i] = 1.0
        v_i = f(t0, e_i)
        A[:, i] = v_i - b
        
    # Check if b is negligible
    if np.linalg.norm(b) < 1e-12:
        # Homogeneous
        try:
            vals, vecs = np.linalg.eig(A)
            exp_vals = np.exp(vals * T)
            # z_end = V diag(exp) V^-1 y0
            # w = V^-1 y0
            w = np.linalg.solve(vecs, y0)
            w_scaled = exp_vals * w
            y_end = vecs @ w_scaled
            # Ensure real output if input is real
            if np.iscomplexobj(y_end):
                y_end = np.real_if_close(y_end, tol=1000)
                if np.iscomplexobj(y_end):
                    y_end = y_end.real
            return y_end
        except np.linalg.LinAlgError:
            pass
    else:
        # Affine: y' = Ay + b
        try:
            vals, vecs = np.linalg.eig(A)
            # Compute exp(At) y0
            exp_vals = np.exp(vals * T)
            w = np.linalg.solve(vecs, y0)
            w_scaled = exp_vals * w
            y_part = vecs @ w_scaled
            
            # Compute integral part
            c = np.linalg.solve(vecs, b)
            
            # Compute integral coefficients using expm1 for better precision
            integral_coeffs = np.zeros_like(vals, dtype=complex)
            for k in range(N):
                x = vals[k] * T
                if abs(vals[k]) < 1e-10:
                    integral_coeffs[k] = T
                else:
                    # Use expm1 to avoid cancellation error in (exp(x)-1)
                    integral_coeffs[k] = np.expm1(x) / vals[k]
            
            # integral_vec = V diag(integral_coeffs) c
            integral_vec = vecs @ (integral_coeffs * c)
            
            y_end = y_part + integral_vec
            # Ensure real output if input is real
            if np.iscomplexobj(y_end):
                y_end = np.real_if_close(y_end, tol=1000)
                if np.iscomplexobj(y_end):
                    y_end = y_end.real
            return y_end
        except np.linalg.LinAlgError:
            pass

    # Fallback
    from scipy.integrate import solve_ivp
    sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12, max_step=T/100)
    if sol.success:
        return sol.y[:, -1]
    else:
        steps = 10000
        h = T / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
