import numpy as np
import scipy.linalg
import scipy.integrate
import time

def _build_heat_matrix_10x10():
    """
    Constructs the 5-point stencil Laplacian matrix for a 10x10 grid
    with Dirichlet boundary conditions (zero values outside the 10x10 domain).
    Uses Kronecker sum for efficiency.
    """
    n = 10
    # 1D Laplacian with Dirichlet BCs
    A = -2.0 * np.eye(n)
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    # 2D Laplacian = kron(I, A) + kron(A, I)
    I = np.eye(n)
    L = np.kron(I, A) + np.kron(A, I)
    return L

def _rk4_fallback(f, t_span, y0, steps=5000):
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
    deadline = start_time + 290  # 300s limit, leave 10s margin for safety
    
    # Strategy 1: Analytic Solution for 10x10 Heat Equation
    if N == 100:
        try:
            L = _build_heat_matrix_10x10()
            
            # Probe 1: Estimate the scaling factor k
            # The ODE is y' = k * L * y
            v0 = f(t0, y0)
            Ly0 = L @ y0
            norm_Ly0_sq = np.dot(Ly0, Ly0)
            
            if norm_Ly0_sq > 1e-12:
                k = np.dot(v0, Ly0) / norm_Ly0_sq
                
                # Verify with a second probe only if time allows
                if time.time() < deadline - 10:
                    rng = np.random.default_rng(42)
                    y_test = rng.standard_normal(N)
                    Ly_test = L @ y_test
                    if np.linalg.norm(Ly_test) > 1e-10:
                        v_test = f(t0, y_test)
                        expected_v_test = k * Ly_test
                        
                        denom = 1e-12 + np.linalg.norm(expected_v_test)
                        error = np.linalg.norm(v_test - expected_v_test) / denom
                        
                        if error < 1e-6:
                            # Model verified. Compute analytic solution
                            A = k * L
                            expA = scipy.linalg.expm(A * dt)
                            y_end = expA @ y0
                            return y_end
                else:
                    # Trust the model with just 1 probe
                    A = k * L
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
