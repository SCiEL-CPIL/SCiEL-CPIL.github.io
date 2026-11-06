import numpy as np
import scipy.linalg
import scipy.integrate
import time

def _build_heat_matrix_10x10():
    """
    Constructs the 5-point stencil Laplacian matrix for a 10x10 grid
    with Dirichlet boundary conditions (zero values outside the 10x10 domain).
    """
    N = 100
    L = np.zeros((N, N))
    for i in range(10):
        for j in range(10):
            idx = i * 10 + j
            L[idx, idx] = -4.0
            if i > 0:
                L[idx, (i - 1) * 10 + j] = 1.0
            if i < 9:
                L[idx, (i + 1) * 10 + j] = 1.0
            if j > 0:
                L[idx, i * 10 + (j - 1)] = 1.0
            if j < 9:
                L[idx, i * 10 + (j + 1)] = 1.0
    return L

def _rk4_fallback(f, t_span, y0, steps=2000):
    h = (t_span[1] - t_span[0]) / steps
    t, y = t_span[0], np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # Handle trivial case
    if np.linalg.norm(y0) < 1e-15:
        return y0.copy()

    start_time = time.time()
    deadline = start_time + 300  # 300s limit, leave margin
    
    # Strategy 1: General Linear Operator Identification
    # Assumes y' = A * y. This covers the heat equation (linear, time-invariant).
    # We identify A by evaluating f at basis vectors.
    if N == 100:
        try:
            A = np.zeros((N, N))
            # Determine columns of A
            for i in range(N):
                e_i = np.zeros(N)
                e_i[i] = 1.0
                A[:, i] = f(t0, e_i)
            
            # Verify linearity and time-independence with random probes
            rng = np.random.default_rng(42)
            valid = True
            for _ in range(5):
                y_rand = rng.standard_normal(N)
                t_rand = t0 + rng.random() * dt
                v_actual = f(t_rand, y_rand)
                v_pred = A @ y_rand
                err = np.linalg.norm(v_actual - v_pred)
                denom = 1e-12 + np.linalg.norm(v_pred)
                if err / denom > 1e-8:
                    valid = False
                    break
            
            if valid:
                # Compute analytic solution
                expA = scipy.linalg.expm(A * dt)
                y_end = expA @ y0
                return y_end
        except Exception:
            pass

    # Strategy 2: SciPy Implicit Solvers
    # These are robust for stiff ODEs like the heat equation.
    # BDF is generally faster than Radau for moderate stiffness.
    
    if time.time() > deadline:
        return _rk4_fallback(f, t_span, y0, 5000)
        
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
        
    if time.time() > deadline:
        return _rk4_fallback(f, t_span, y0, 5000)

    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Strategy 3: Fallback RK4
    # Use a high number of steps to ensure stability and accuracy
    # if implicit solvers fail (unlikely).
    # Note: RK4 might be unstable if h is too large for the stiffness.
    # For 10x10 heat eq, max eigenvalue lambda ~ -8/h^2.
    # Stability limit h < 2/|lambda| ~ h_grid^2.
    # If grid is 1/11, h_grid^2 ~ 0.008.
    # dt = 0.05. Steps ~ 7 to be stable.
    # For accuracy, we need many more.
    # 5000 steps -> h ~ 1e-5. Safe.
    
    return _rk4_fallback(f, t_span, y0, 5000)
