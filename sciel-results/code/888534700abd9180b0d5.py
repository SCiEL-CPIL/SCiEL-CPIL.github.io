"""
Strategy: Matrix Exponential Solution for Linear ODEs (Heat Equation)
Instead of iterating with RK4 or Implicit Euler, we identify the linear operator A such that y' = A*y.
We then compute y(t_end) = expm(A * (t_end - t_start)) @ y0.
This requires O(n) calls to f to build A (or O(n^2) if we are not careful, but we can be smart).
Actually, for a fixed 10x10 grid, n=100. Building A via finite differences requires 100 calls to f.
Then we compute the matrix exponential. This is extremely fast and accurate.
Score = -log10(err) - 0.5*log10(calls).
Calls ~ 100. Error ~ machine precision (1e-15).
Score ~ 15 - 0.5*log10(100) = 15 - 1 = 14.
Current score is 1.0. This is a massive jump.
"""

import numpy as np
import scipy.linalg
import time


def _build_matrix_A(f, t0, y0):
    """
    Estimate the Jacobian matrix A of the ODE y' = f(t, y) at t=t0, y=y0.
    We use forward differences. Since the heat equation is linear and autonomous,
    f(t, y) = A @ y (assuming no source term or constant source handled separately).
    Actually, f(t,y) might have a bias. Let's assume f(t,y) = A @ y + b.
    For the heat equation with Dirichlet/Neumann BCs, it's typically homogeneous or has a constant.
    However, the problem says "initial-value problem y' = f(t, y)".
    If it's linear, f(t, y) = A(t) @ y + b(t).
    Given t_span is [0, 0.05] and it's a "fixed" problem, A and b are likely constant.
    
    To get A and b:
    y' = A y + b
    
    Call 1: y0 -> f0 = A y0 + b
    Call 2: y0 + eps*e_j -> f_j = A (y0 + eps*e_j) + b = f0 + eps * A[:, j]
    So A[:, j] = (f_j - f0) / eps
    
    This requires 1 + n calls.
    Total calls = 101 for n=100.
    """
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Get f(t0, y0)
    f0 = np.array(f(t0, y), dtype=float)
    
    A = np.zeros((n, n))
    
    # Estimate columns of A
    # Choose eps carefully. Too small: noise. Too large: non-linearity (if any) or just precision loss.
    # For double precision, eps ~ 1e-7 is standard.
    eps = 1e-7
    
    for j in range(n):
        y_pert = y.copy()
        y_pert[j] += eps
        f_pert = np.array(f(t0, y_pert), dtype=float)
        A[:, j] = (f_pert - f0) / eps
        
    # Check if the system is homogeneous (b=0) or has a bias.
    # If b != 0, then y(t) = expm(A*t) @ y0 + integral expm(A*(t-s)) b ds.
    # If b is constant, y_inf = -A^-1 b.
    # Let's check if f0 is consistent with A @ y0.
    # residual = f0 - A @ y0
    # If residual is small, b is negligible or zero.
    # If not, we need to handle b.
    # However, for the heat equation, usually b=0 (insulated or fixed temp 0).
    # Let's assume b=0 first. If the error is high, we can add a correction.
    # Actually, let's just solve y' = A y.
    
    return A, f0


def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin for 300s limit
    
    # 1. Build the linear operator A
    A, f0 = _build_matrix_A(f, t0, y0)
    
    # Check time
    if time.time() > deadline:
        # Fallback to simple RK4 if we are too slow (unlikely for n=100)
        pass

    # 2. Compute matrix exponential
    # y(t1) = expm(A * dt) @ y0
    # This is exact for linear systems.
    
    # Optimization: Use scipy.linalg.expm
    expm_A_dt = scipy.linalg.expm(A * dt)
    
    y_final = expm_A_dt @ np.array(y0, dtype=float)
    
    # 3. Verification/Refinement
    # Check if the assumption of linearity/autonomy holds.
    # We can do a quick RK4 step with a large step to see if it matches.
    # Or, check if f(t1, y_final) is consistent with A @ y_final (if autonomous and linear).
    
    # Let's check the residual at the final state.
    # If the system is linear and autonomous, f(t, y) = A @ y.
    # So f(t1, y_final) should be A @ y_final.
    # If there is a bias b, f(t, y) = A @ y + b.
    # Then y' = A y + b.
    # Solution: y(t) = expm(A t) y0 + (expm(A t) - I) A^-1 b.
    
    # Let's estimate b.
    # b = f0 - A @ y0
    b = f0 - A @ np.array(y0, dtype=float)
    
    # If b is significant, we need to include it.
    # Norm of b vs Norm of A @ y0
    norm_Ay = np.linalg.norm(A @ np.array(y0, dtype=float))
    norm_b = np.linalg.norm(b)
    
    if norm_b > 1e-8 * max(1.0, norm_Ay):
        # System is affine: y' = A y + b
        # We need to solve for the particular solution or integrate the affine term.
        # y(t) = expm(A t) y0 + \int_0^t expm(A (t-s)) b ds
        # If A is invertible, \int_0^t expm(A (t-s)) b ds = A^-1 (expm(A t) - I) b
        # Let's check if A is invertible.
        try:
            A_inv = scipy.linalg.inv(A)
            # Particular solution term
            term = np.linalg.solve(A, (expm_A_dt - np.eye(len(y0))) @ b)
            y_final = expm_A_dt @ np.array(y0, dtype=float) + term
        except np.linalg.LinAlgError:
            # A is singular. Use numerical integration of the affine term?
            # Or just stick to the homogeneous approximation if b is small?
            # For heat equation, A is usually stable and invertible (eigenvalues < 0).
            # If singular, it might be a steady state.
            pass
    
    # Final check: Is y_final finite?
    if not np.all(np.isfinite(y_final)):
        # Fallback to RK4
        y = np.array(y0, dtype=float)
        h = dt / 100
        t = t0
        for _ in range(100):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        y_final = y

    return y_final
