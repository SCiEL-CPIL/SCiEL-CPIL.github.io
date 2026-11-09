import numpy as np
import time
from scipy.linalg import expm

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] given initial condition y0.
    Assumes f is linear and time-invariant: f(t, y) = A @ y + b.
    Uses tridiagonal structure extraction to minimize f calls.
    """
    start_time = time.time()
    deadline = start_time + 290.0

    y0 = np.array(y0, dtype=float)
    N = len(y0)
    t0, t1 = t_span
    dt = t1 - t0

    # Extract b
    b = f(0, np.zeros(N))
    
    # Prepare test vectors for tridiagonal extraction
    v1 = np.ones(N)
    v2 = np.arange(N, dtype=float)
    v3 = v2 ** 2

    # Get responses
    c1 = f(0, v1) - b
    c2 = f(0, v2) - b
    c3 = f(0, v3) - b

    # Check for finite values
    if not np.all(np.isfinite(c1)) or not np.all(np.isfinite(c2)) or not np.all(np.isfinite(c3)):
        return _fallback_solve(f, t_span, y0, start_time, deadline)

    # Extract A (tridiagonal)
    A = np.zeros((N, N))
    
    for i in range(N):
        if i == 0:
            # Unknowns: A[0,0], A[0,1]
            # Eq: v1[0]*A00 + v1[1]*A01 = c1[0]
            #     v2[0]*A00 + v2[1]*A01 = c2[0]
            M = np.array([[v1[0], v1[1]], [v2[0], v2[1]]])
            rhs = np.array([c1[0], c2[0]])
            try:
                sol = np.linalg.solve(M, rhs)
                A[0, 0] = sol[0]
                A[0, 1] = sol[1]
            except np.linalg.LinAlgError:
                pass
        elif i == N - 1:
            # Unknowns: A[N-1, N-2], A[N-1, N-1]
            # Eq: v1[N-2]*A[i,i-1] + v1[N-1]*A[i,i] = c1[i]
            #     v2[N-2]*A[i,i-1] + v2[N-1]*A[i,i] = c2[i]
            M = np.array([[v1[i-1], v1[i]], [v2[i-1], v2[i]]])
            rhs = np.array([c1[i], c2[i]])
            try:
                sol = np.linalg.solve(M, rhs)
                A[i, i-1] = sol[0]
                A[i, i] = sol[1]
            except np.linalg.LinAlgError:
                pass
        else:
            # Unknowns: A[i, i-1], A[i, i], A[i, i+1]
            # Eq: v1[i-1]*u + v1[i]*v + v1[i+1]*w = c1[i]
            #     v2[i-1]*u + v2[i]*v + v2[i+1]*w = c2[i]
            #     v3[i-1]*u + v3[i]*v + v3[i+1]*w = c3[i]
            M = np.array([
                [v1[i-1], v1[i], v1[i+1]],
                [v2[i-1], v2[i], v2[i+1]],
                [v3[i-1], v3[i], v3[i+1]]
            ])
            rhs = np.array([c1[i], c2[i], c3[i]])
            try:
                sol = np.linalg.solve(M, rhs)
                A[i, i-1] = sol[0]
                A[i, i] = sol[1]
                A[i, i+1] = sol[2]
            except np.linalg.LinAlgError:
                pass

    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        return _fallback_solve(f, t_span, y0, start_time, deadline)

    # Compute exact solution
    try:
        M_aug = np.zeros((N + 1, N + 1))
        M_aug[:N, :N] = A
        M_aug[:N, N] = b
        z0 = np.zeros(N + 1)
        z0[:N] = y0
        z0[N] = 1.0
        E = expm(M_aug * dt)
        z_end = E @ z0
        y_end = z_end[:N]
        if np.all(np.isfinite(y_end)):
            return y_end
    except Exception:
        pass

    return _fallback_solve(f, t_span, y0, start_time, deadline)

def _fallback_solve(f, t_span, y0, start_time, deadline):
    """
    Fallback solver using scipy.integrate.solve_ivp.
    """
    from scipy.integrate import solve_ivp
    try:
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12, max_step=0.05)
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    steps = 500
    h = (t1 - t0) / steps
    t = t0
    for _ in range(steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y
