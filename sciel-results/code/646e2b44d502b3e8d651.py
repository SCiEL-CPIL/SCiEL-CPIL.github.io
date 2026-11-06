import numpy as np
import scipy.linalg
import scipy.integrate

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # 1. Build Jacobian A assuming linearity
    # f(t, y) = A y + b
    # f(t, e_j) = A e_j + b = A[:, j] + b
    # So A[:, j] = f(t, e_j) - b
    # We don't know b yet.
    # Let's get f(t, e_j) for all j.
    F_basis = np.zeros((N, N))
    for j in range(N):
        e_j = np.zeros(N)
        e_j[j] = 1.0
        F_basis[:, j] = f(t0, e_j)
        
    # Assume homogeneous first (b=0)
    # Then A = F_basis
    A_hom = F_basis
    
    # Check if f(t, y0) matches A_hom @ y0
    y_act = f(t0, y0)
    y_pred_hom = A_hom @ y0
    
    norm_diff_hom = np.linalg.norm(y_act - y_pred_hom)
    norm_scale = 1.0 + np.linalg.norm(y_act)
    
    dt = t1 - t0

    if norm_diff_hom < 1e-8 * norm_scale:
        # Homogeneous linear system
        try:
            # Check for symmetry to use more efficient/accurate eigh
            if np.allclose(A_hom, A_hom.T, rtol=1e-10, atol=1e-12):
                w, V = scipy.linalg.eigh(A_hom)
                exp_w = np.exp(w * dt)
                # y_end = V @ (exp_w * (V.T @ y0))
                y_end = V @ (exp_w * (V.T @ y0))
            else:
                y_end = scipy.linalg.expm_multiply(A_hom * dt, y0)
            return y_end
        except Exception:
            pass # Fallback

    # Check inhomogeneous linear system
    # Need b = f(t, 0)
    b = f(t0, np.zeros(N))
    
    # If f(t, y) = A y + b, then f(t, e_j) = A e_j + b
    # So A = F_basis - b
    A_inh = F_basis - b[:, np.newaxis]
    
    y_pred_inh = A_inh @ y0 + b
    
    norm_diff_inh = np.linalg.norm(y_act - y_pred_inh)
    
    if norm_diff_inh < 1e-8 * norm_scale:
        # Inhomogeneous linear system
        try:
            # Compute expm(A * dt)
            if np.allclose(A_inh, A_inh.T, rtol=1e-10, atol=1e-12):
                w, V = scipy.linalg.eigh(A_inh)
                exp_w = np.exp(w * dt)
                M = V @ np.diag(exp_w) @ V.T
            else:
                M = scipy.linalg.expm(A_inh * dt)
            
            y_end = M @ y0
            if np.linalg.norm(b) > 1e-10:
                try:
                    x = scipy.linalg.solve(A_inh, b)
                    y_end += (M - np.eye(N)) @ x
                except np.linalg.LinAlgError:
                    x = scipy.linalg.pinv(A_inh) @ b
                    y_end += (M - np.eye(N)) @ x
            return y_end
        except Exception:
            pass # Fallback

    # Fallback: Stiff ODE Solver
    sol = scipy.integrate.solve_ivp(
        f, t_span, y0, 
        method='Radau', 
        rtol=1e-12, 
        atol=1e-14
    )
    if sol.success:
        return sol.y[:, -1]
    else:
        sol2 = scipy.integrate.solve_ivp(
            f, t_span, y0, 
            method='BDF', 
            rtol=1e-12, 
            atol=1e-14
        )
        return sol2.y[:, -1]
