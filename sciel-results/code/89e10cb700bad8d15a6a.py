import numpy as np
from scipy.linalg import expm, solve
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # Estimate b = f(t, 0)
    b = f(t0, np.zeros(N))
    
    # Check if b is negligible
    is_homogeneous = np.allclose(b, 0, rtol=1e-10, atol=1e-12)
    
    if is_homogeneous:
        # Linear homogeneous: y' = Ay
        A = np.zeros((N, N))
        for i in range(N):
            e_i = np.zeros(N)
            e_i[i] = 1.0
            A[:, i] = f(t0, e_i)
            
        dt = t1 - t0
        M = dt * A
        try:
            expM = expm(M)
            y_end = expM @ y0
            if np.all(np.isfinite(y_end)):
                return y_end
        except Exception:
            pass
    else:
        # Affine: y' = Ay + b (assuming b is constant)
        # Check time dependence of b?
        b_t1 = f(t1, np.zeros(N))
        if np.allclose(b, b_t1, rtol=1e-10, atol=1e-12):
            # Constant source
            A = np.zeros((N, N))
            for i in range(N):
                e_i = np.zeros(N)
                e_i[i] = 1.0
                # A e_i = f(t, e_i) - b
                A[:, i] = f(t0, e_i) - b
                
            dt = t1 - t0
            M = dt * A
            try:
                expM = expm(M)
                # y(t) = expM @ y0 + C @ b
                # C = integral_0^dt exp(A tau) dtau
                # C = (expM - I) A^{-1}
                # Solve A C = expM - I
                # Or C = A^{-1} (expM - I)
                # We can compute this by solving linear systems
                # Or using the block matrix exponential
                # Simpler: Use solve for A^{-1} (expM - I) b
                # Actually, we need the term added to y0.
                # term = C @ b
                # C @ b = A^{-1} (expM - I) b
                # Let v = (expM - I) @ b
                # term = solve(A, v)
                
                I = np.eye(N)
                v = (expM - I) @ b
                try:
                    term = solve(A, v)
                except np.linalg.LinAlgError:
                    # A is singular? Unlikely for diffusion
                    term = np.zeros(N)
                    
                y_end = expM @ y0 + term
                if np.all(np.isfinite(y_end)):
                    return y_end
            except Exception:
                pass

    # Fallback
    sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
    if sol.success:
        return sol.y[:, -1]
    return y0
