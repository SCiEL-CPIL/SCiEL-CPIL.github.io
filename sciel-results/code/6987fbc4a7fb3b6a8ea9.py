"""ODE solver for convection-diffusion 100-point problem using eigendecomposition strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Safety deadline guard
    start_time = time.time()
    deadline = start_time + 290.0
    
    # 1. Detect linearity and time-invariance, and extract A, b
    # For y' = f(t, y), if linear and time-invariant: f(t, y) = A @ y + b
    # We can determine A and b by calling f at specific points.
    
    y0_test = np.zeros(n)
    b = f(t0, y0_test)
    
    # Extract A columns
    A = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        # f(t, ej) = A @ ej + b = A[:, j] + b
        # So A[:, j] = f(t, ej) - b
        A[:, j] = f(t0, ej) - b
        
    # Verify linearity with a random vector
    y_rand = np.random.randn(n)
    y_rand[0] = 1.0  # Ensure non-zero
    f_pred = A @ y_rand + b
    f_actual = f(t0, y_rand)
    
    if not np.allclose(f_pred, f_actual, rtol=1e-10, atol=1e-12):
        # Fallback to RK4 if not linear
        return _rk4_fallback(f, t_span, y0)
    
    # 2. Solve y' = A y + b using eigendecomposition
    # The solution is y(t) = expm(A*dt) @ y0 + z where z' = A*z + b, z(0) = 0
    # z(dt) = integral_0^dt expm(A*tau) b dtau
    # If A = V diag(L) V^-1, then expm(A*dt) = V diag(exp(L*dt)) V^-1
    # And integral_0^dt expm(A*tau) b dtau = V diag((exp(L*dt) - 1) / L) V^-1 b (with limit 1 for L=0)
    
    dt = t1 - t0
    
    # Compute eigendecomposition of A
    # Use np.linalg.eig for general matrices
    try:
        eigenvalues, V = np.linalg.eig(A)
    except np.linalg.LinAlgError:
        # Fallback to expm if eig fails
        return _expm_fallback(f, t_span, y0)
    
    # Check if eigenvalues are real (or close to real)
    # If complex, we might need to handle carefully, but for convection-diffusion, they should be real or complex conjugate pairs.
    # np.linalg.eig handles complex eigenvalues. The formula still works.
    
    # Compute V^-1
    try:
        V_inv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        # If V is ill-conditioned, fall back to expm
        return _expm_fallback(f, t_span, y0)
    
    # Compute exp(L * dt)
    exp_eig = np.exp(eigenvalues * dt)
    
    # Compute the particular solution factor
    # (exp(L * dt) - 1) / L
    # Handle L = 0 case
    L = eigenvalues
    factor = np.ones_like(L, dtype=complex)
    nonzero = np.abs(L) > 1e-12
    factor[nonzero] = (exp_eig[nonzero] - 1) / L[nonzero]
    factor[~nonzero] = dt  # Limit as L->0 of (e^(L*dt)-1)/L is dt
    
    # Compute homogeneous part: expm(A*dt) @ y0
    # expm(A*dt) = V @ diag(exp_eig) @ V_inv
    E = V @ np.diag(exp_eig) @ V_inv
    y_hom = E @ y
    
    # Compute particular part: V @ diag(factor) @ V_inv @ b
    # Let's compute V_inv @ b first to save some operations
    V_inv_b = V_inv @ b
    particular_factor = factor * V_inv_b
    y_part = V @ particular_factor
    
    y_final = y_hom + y_part
    
    # Check time limit
    if time.time() > deadline:
        return y_final.real
    
    return y_final.real

def _rk4_fallback(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    h = (t1 - t0) / 1000
    t = t0
    for _ in range(1000):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y

def _expm_fallback(f, t_span, y0):
    from scipy.linalg import expm
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    y0_test = np.zeros(n)
    b = f(t0, y0_test)
    
    A = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        A[:, j] = f(t0, ej) - b
        
    dt = t1 - t0
    
    A_aug = np.zeros((n + 1, n + 1))
    A_aug[:n, :n] = A
    A_aug[:n, n] = b
    
    w0 = np.zeros(n + 1)
    w0[n] = 1.0
    
    E_aug = expm(A_aug * dt)
    w_final = E_aug @ w0
    
    z = w_final[:n]
    E = E_aug[:n, :n]
    
    y_final = E @ y + z
    return y_final
# EVOLVE-BLOCK-END
