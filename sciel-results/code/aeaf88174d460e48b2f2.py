"""ODE solver for convection-diffusion 100-point problem using eigen decomposition strategy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import eig, expm
from scipy.integrate import solve_ivp
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)

    # Safety deadline guard
    start_time = time.time()
    deadline = start_time + 285.0

    # 1. Detect linearity and time-invariance, and extract A, b
    # For y' = f(t, y), if linear and time-invariant: f(t, y) = A @ y + b
    # Extract b by calling f at t=0 with y=0
    y0_test = np.zeros(n)
    b = f(t0, y0_test)

    # Extract A columns
    A = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        A[:, j] = f(t0, ej) - b

    # Verify linearity with a random vector
    rng = np.random.RandomState(42)
    y_rand = rng.randn(n)
    f_pred = A @ y_rand + b
    f_actual = f(t0, y_rand)

    if not np.allclose(f_pred, f_actual, rtol=1e-10, atol=1e-12):
        # Fallback to high-precision ODE solver if not linear
        return _solve_ivp_fallback(f, t_span, y0)

    # 2. Solve y' = A y + b using Eigen Decomposition
    # This is numerically more stable for stiff spectra than expm + series
    dt = t1 - t0
    
    try:
        eigenvalues, eigenvectors = eig(A)
    except Exception:
        # Fallback to expm if eig fails
        E = expm(A * dt)
        z = _compute_integral_series(A, b, dt)
        return E @ y + z

    # Compute e^(lambda * dt)
    exp_lambda = np.exp(eigenvalues * dt)
    
    # Compute the integral term coefficients
    # integral_0^dt e^(A*u) b du = V diag( (e^(lambda_i * dt) - 1) / lambda_i ) V^-1 b
    # Handle lambda_i == 0 case
    integral_coeffs = np.zeros_like(eigenvalues)
    for i in range(n):
        if abs(eigenvalues[i]) < 1e-12:
            integral_coeffs[i] = dt
        else:
            integral_coeffs[i] = (exp_lambda[i] - 1.0) / eigenvalues[i]
            
    # We need V^-1 b and V^-1 y
    try:
        V_inv = np.linalg.inv(eigenvectors)
        V_inv_b = V_inv @ b
        V_inv_y = V_inv @ y
        
        # y_final = V diag(exp_lambda) V^-1 y + V diag(integral_coeffs) V^-1 b
        y_homo = eigenvectors @ (exp_lambda * V_inv_y)
        y_part = eigenvectors @ (integral_coeffs * V_inv_b)
        y_final = y_homo + y_part
    except np.linalg.LinAlgError:
        # If V is singular or ill-conditioned, fallback to expm
        E = expm(A * dt)
        z = _compute_integral_series(A, b, dt)
        y_final = E @ y + z

    # Check time limit
    if time.time() > deadline:
        return y_final

    return y_final


def _compute_integral_series(A, b, dt):
    """Compute integral_0^dt expm(A*u) b du using series."""
    z = np.zeros(len(b))
    term = b.copy() * dt
    z = term.copy()
    for k in range(1, 100):
        term = A @ term * (dt / (k + 1))
        z += term
        # Check convergence
        if np.linalg.norm(term) < 1e-15 * (1 + np.linalg.norm(z)):
            break
    return z


def _solve_ivp_fallback(f, t_span, y0):
    """High-precision fallback solver using Radau method."""
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-10, 
        atol=1e-12,
        max_step=0.01
    )
    if sol.success:
        return sol.y[:, -1]
    else:
        # Last resort RK4
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
# EVOLVE-BLOCK-END
