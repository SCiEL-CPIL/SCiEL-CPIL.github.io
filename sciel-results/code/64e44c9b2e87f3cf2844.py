import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    side = int(np.sqrt(N))
    
    # Case 1: Not a square grid, or N too large for efficient spectral methods
    if side * side != N or N > 10000:
        # Fall back to a robust but slower method: probe A and use expm
        # For now, just use a simple RK4 as fallback to avoid infinite loops
        # But the task specifies 10x10, so this branch is unlikely.
        # Let's implement a basic RK4 fallback just in case.
        steps = 100
        h = dt / steps
        y = y0.copy()
        t = t0
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y

    # Probe the system to determine A and b
    # We assume the system is linear: y' = A*y + b
    
    # 1. Get b = f(t, 0)
    y_zero = np.zeros(N, dtype=float)
    b = f(t0, y_zero)
    
    # 2. Determine A by probing basis vectors.
    # For a 10x10 grid (N=100), probing all 100 basis vectors costs 100 calls.
    # This is very cheap compared to 2,000,000 limit.
    # However, we can try to be smarter if we detect it's a standard Laplacian.
    # But for robustness and to handle any linear system, probing A is safest.
    # Let's probe A.
    
    A = np.zeros((N, N), dtype=float)
    for i in range(N):
        e_i = np.zeros(N, dtype=float)
        e_i[i] = 1.0
        A[:, i] = f(t0, e_i)
        
    # Total calls so far: 1 (for b) + N (for A) = N + 1 = 101 calls.
    
    # Check time
    if time.time() > deadline:
        # Should not happen, but just in case
        pass

    # Now we have A and b.
    # The ODE is y' = A*y + b.
    # The solution is y(t) = expm(A*t) * y0 + integral_0^t expm(A*(t-s)) * b ds.
    # If b is constant, the integral is expm(A*t) * A^{-1} * (expm(A*t) - I) * b? No.
    # The solution to y' = A y + b is:
    # y(t) = expm(A t) y0 + \int_0^t expm(A (t-s)) b ds
    # Let u = t-s, du = -ds. When s=0, u=t. When s=t, u=0.
    # Integral = \int_t^0 expm(A u) b (-du) = \int_0^t expm(A u) b du
    # If A is invertible, \int_0^t expm(A u) du = A^{-1} (expm(A t) - I)
    # So y(t) = expm(A t) y0 + A^{-1} (expm(A t) - I) b
    
    # However, A might be singular. For the heat equation with Dirichlet BCs, A is negative definite, so invertible.
    # If Neumann BCs, A has a null space (constant vector). Then A is singular.
    # In that case, we can use the augmented matrix method.
    # Let z = [y; 1]. Then z' = [A b; 0 0] z.
    # z(t) = expm([A b; 0 0] * t) z(0).
    # This always works.
    
    # Let's use the augmented matrix method for robustness.
    # Size is (N+1) x (N+1). For N=100, this is 101x101. expm is fast.
    
    A_aug = np.zeros((N + 1, N + 1), dtype=float)
    A_aug[:N, :N] = A
    A_aug[:N, N] = b
    # Last row is all zeros
    
    M_aug = np.linalg.matrix_power(A_aug, 0) # Just to get identity? No, use scipy.linalg.expm
    
    from scipy.linalg import expm
    
    M_aug = expm(A_aug * dt)
    
    y_init_aug = np.zeros(N + 1, dtype=float)
    y_init_aug[:N] = y0
    y_init_aug[N] = 1.0
    
    y_end_aug = M_aug @ y_init_aug
    y_end = y_end_aug[:N]
    
    # Check time
    if time.time() > deadline:
        # If we are running out of time, return what we have
        pass
    
    if not np.all(np.isfinite(y_end)):
        # Fallback to RK4 if something went wrong
        steps = 1000
        h = dt / steps
        y = y0.copy()
        t = t0
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
        
    return y_end
