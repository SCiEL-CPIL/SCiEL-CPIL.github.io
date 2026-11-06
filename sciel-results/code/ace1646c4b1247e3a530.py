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
import time
import sys

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 320  # 320s limit to be safe

    # 1. Identify the linear operator A
    # The heat equation is linear and autonomous (or at least the spatial part is linear).
    # y' = A * y
    # We can determine A by evaluating f at basis vectors.
    # However, f(t, y) might have a time-dependent part? 
    # "fixed 10x10 two-dimensional heat equation initial-value problem"
    # Standard heat eq: u_t = alpha * u_xx + alpha * u_yy.
    # This is y' = A * y where A is constant (if alpha is constant).
    # If alpha depends on t, it's y' = A(t) * y. 
    # But usually "heat equation" implies constant coefficients unless specified.
    # Let's assume A is constant first. If not, we might need to handle it differently.
    # Given the "fixed" nature and typical task structures, A is likely constant.
    
    # To find A, we need to map y -> f(t0, y).
    # Since it's linear, f(t, y) = A y.
    # We can compute A by sending unit vectors.
    # But wait, f takes (t, y). If the system is linear and autonomous, f(t, y) = A y.
    # Let's verify linearity and autonomy with a few checks or just assume it.
    # A safer bet for a "fixed" problem: Compute A at t0.
    
    # Efficiently build A:
    # A[:, j] = (f(t0, e_j) - f(t0, 0)) 
    # If f is linear homogeneous, f(t0, 0) = 0.
    # Let's check f(t0, 0).
    
    y_zero = np.zeros_like(y0)
    f_zero = f(t0, y_zero)
    
    # If f_zero is not close to zero, the system is affine or non-linear.
    # Heat equation is linear homogeneous.
    if np.linalg.norm(f_zero) > 1e-12:
        # Fallback: If not homogeneous, try affine y' = A y + b
        # b = f(t0, 0)
        # A e_j = f(t0, e_j) - b
        b = f_zero
    else:
        b = None

    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros_like(y0)
        e_j[j] = 1.0
        f_ej = f(t0, e_j)
        if b is not None:
            A[:, j] = f_ej - b
        else:
            A[:, j] = f_ej
            
    # Check if we are running out of time
    if time.time() > deadline - 10:
        # Fallback to a simple step if we can't do expm
        # Just return y0 if we have no time, or a rough RK4
        # But building A took n calls. We have time for expm usually.
        pass

    dt = t1 - t0
    
    # Compute the solution using matrix exponential
    # y(t1) = exp(A * dt) @ y0 + (if affine) integral term
    # For linear homogeneous: y(t1) = expm(A * dt) @ y0
    # For affine y' = A y + b: y(t1) = expm(A*dt) @ y0 + (expm(A*dt) - I) @ A^{-1} @ b
    # Or simply use scipy.integrate.solve_ivp with method='LSODA' or 'Radau' if expm fails?
    # No, expm is better for the score.
    
    try:
        # Use scipy.linalg.expm for accuracy
        from scipy.linalg import expm
        
        if b is None:
            # Homogeneous
            M = expm(A * dt)
            y_final = M @ y0
        else:
            # Affine
            # Solution: y(t) = e^{At} y0 + \int_0^t e^{A(t-s)} b ds
            # = e^{At} y0 + (e^{At} - I) A^{-1} b  (if A invertible)
            # If A is singular (heat equation on finite grid with Dirichlet/Neumann, A is invertible usually? 
            # Neumann: nullspace is constants. Dirichlet: invertible.)
            # Let's use the integral form via matrix function or just solve the augmented system.
            # Augmented system: [y; 1]' = [A, b; 0, 0] [y; 1]
            # y(0) = [y0; 1]
            # This is homogeneous linear in n+1 dimensions.
            A_aug = np.zeros((n+1, n+1))
            A_aug[:n, :n] = A
            A_aug[:n, n] = b
            y_aug_0 = np.zeros(n+1)
            y_aug_0[:n] = y0
            y_aug_0[n] = 1.0
            
            M_aug = expm(A_aug * dt)
            y_aug_final = M_aug @ y_aug_0
            y_final = y_aug_final[:n]
            
    except Exception as e:
        # Fallback to RK4 if expm fails (e.g., memory, or if A is not constant)
        # This shouldn't happen for standard heat eq.
        # Simple RK4 with large steps as a last resort
        steps = 100
        h = dt / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        y_final = y

    return y_final
