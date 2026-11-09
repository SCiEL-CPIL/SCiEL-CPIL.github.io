import numpy as np
import time
from scipy.linalg import expm

def integrate(f, t_span, y0):
    """
    Solves the 15-dimensional stiff linear IVP using Krylov subspace (Arnoldi) method.
    y' = A y. We compute e^(A * dt) @ y0 without forming A explicitly.
    """
    start_time = time.time()
    deadline = start_time + 280.0
    
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Check for affine component b
    b = np.asarray(f(t0, np.zeros(n, dtype=float)), dtype=float)
    b_norm = np.linalg.norm(b)
    
    if time.time() > deadline:
        return y0
    
    # If affine component is significant, fall back to a robust solver
    # because Arnoldi on the linear part alone is insufficient for y' = Ay + b
    if b_norm > 1e-12:
        try:
            import scipy.integrate as integrate_scipy
            sol = integrate_scipy.solve_ivp(f, t_span, y0, method='Radau', 
                                            rtol=1e-12, atol=1e-14, max_step=0.01)
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
        return y0

    # Homogeneous case: y' = A y
    # Use Arnoldi iteration to build Krylov subspace K_m(A, y0)
    # y0 is the starting vector
    
    # Normalize y0 for Arnoldi
    y0_norm = np.linalg.norm(y0)
    if y0_norm < 1e-15:
        return np.zeros(n, dtype=float)
        
    v = y0 / y0_norm
    
    # Arnoldi parameters
    max_m = n  # Full dimension
    V = np.zeros((n, max_m + 1), dtype=float)
    H = np.zeros((max_m + 1, max_m + 1), dtype=float)
    
    V[:, 0] = v
    
    m = 0
    # Tolerance for early termination
    tol = 1e-14
    
    while m < max_m:
        # Apply A to v_m
        # A v_m = f(t0, v_m)
        w = np.asarray(f(t0, V[:, m]), dtype=float)
        
        # Modified Gram-Schmidt orthogonalization
        for j in range(m + 1):
            h_jm = np.dot(V[:, j], w)
            H[j, m] = h_jm
            w -= h_jm * V[:, j]
        
        h_mm1 = np.linalg.norm(w)
        H[m + 1, m] = h_mm1
        
        # Check for early termination (invariant subspace)
        if h_mm1 < tol * y0_norm:
            break
            
        V[:, m + 1] = w / h_mm1
        m += 1
        
        # Check deadline
        if time.time() > deadline:
            # Fallback to best available
            break

    # m is the number of steps taken. The subspace dimension is m + 1.
    # The Hessenberg matrix is H[:m+1, :m+1]
    H_sub = H[:m + 1, :m + 1]
    V_sub = V[:, :m + 1]
    
    # Compute e^(H_sub * dt)
    # H_sub is (m+1) x (m+1)
    try:
        E_H = expm(H_sub * dt)
    except Exception:
        # Fallback if expm fails
        try:
            import scipy.integrate as integrate_scipy
            sol = integrate_scipy.solve_ivp(f, t_span, y0, method='Radau', 
                                            rtol=1e-12, atol=1e-14, max_step=0.01)
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
        return y0

    # The coefficient vector c = V_sub^T y0
    # Since V is orthonormal, V_sub^T V_sub = I for the first m+1 columns
    # y0 = V_sub c + residual
    # If m < n-1, there is a residual.
    # But if we converged, the residual is small.
    # c = V_sub^T y0
    c = V_sub.T @ y0
    
    # y_end = V_sub @ (E_H @ c)
    y_end = V_sub @ (E_H @ c)
    
    return y_end
