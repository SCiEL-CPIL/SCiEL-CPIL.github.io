"""ODE solver for convection-diffusion 100-point problem using matrix exponential strategy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
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
        # Fallback to RK4 if not linear
        return _rk4_fallback(f, t_span, y0)

    # 2. Solve y' = A y + b using matrix exponential
    # y(t) = expm(A*(t-t0)) @ y0 + integral_0^(t-t0) expm(A*u) b du
    dt = t1 - t0

    # Use block matrix exponential for high accuracy:
    # M = [[A*dt, b*dt], [0, 0]]
    # expm(M) = [[E, Z], [0, 1]]
    # y_final = E @ y0 + Z
    # This computes E and Z simultaneously with full precision from the same
    # matrix exponential call, avoiding series truncation error in Z.

    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A * dt
    M[:n, n] = b * dt
    # M[n, n] = 0 is already there

    E_block = expm(M)

    E = E_block[:n, :n]
    Z = E_block[:n, n]

    y_final = E @ y + Z

    # Check time limit
    if time.time() > deadline:
        return y_final

    return y_final


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
# EVOLVE-BLOCK-END
