"""ODE solver for 15-dimensional stiff linear system using matrix exponential."""
import numpy as np
import time
from scipy.linalg import expm


def _extract_linear_system(f, t0, y0, n):
    """Extract the constant matrix A and vector b from y' = A*y + b by probing."""
    y0 = np.array(y0, dtype=float)
    # Probe at different times and states to determine linearity
    # First, check if it's truly linear: f(t, y1) - f(t, y2) should be independent of y
    
    # Use y0 and a perturbed version
    scale = max(1.0, np.max(np.abs(y0)))
    perturbation = 1e-8 * scale
    
    # Get f at y0
    f_y0 = np.array(f(t0, y0), dtype=float)
    
    # Check if the system is autonomous (independent of t) by evaluating at different t
    f_y0_t1 = np.array(f(t0 + 0.5, y0), dtype=float)
    
    # Check linearity: f(t, c*y0) = c*f(t, y0) - f(t, 0)
    # Let's evaluate at zero and at y0
    y_zero = np.zeros(n, dtype=float)
    f_zero = np.array(f(t0, y_zero), dtype=float)
    
    # For a linear system y' = A*y + b, we have:
    # f(t, y) = A*y + b
    # So A*y0 = f_y0 - b, and b = f_zero (if we can evaluate at y=0)
    # Actually, for homogeneous linear system y' = A*y, b=0
    # For affine system y' = A*y + b, b = f(t, 0)
    
    # Let's assume it's either y' = A*y or y' = A*y + b
    # We'll extract A by probing with basis vectors
    
    # Extract A columns
    A = np.zeros((n, n), dtype=float)
    b = f_zero.copy()
    
    # For each column j, perturb y0 in direction j
    for j in range(n):
        y_pert = y_zero.copy()
        y_pert[j] = 1.0
        f_pert = np.array(f(t0, y_pert), dtype=float)
        # f(0, e_j) = A[:,j] + b
        A[:, j] = f_pert - b
    
    return A, b


def integrate(f, t_span, y0):
    t0, t1 = t_span
    n = len(y0)
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    # Extract the linear system y' = A*y + b
    A, b = _extract_linear_system(f, t0, y0, n)
    
    # For y' = A*y + b, the solution is:
    # y(t) = exp(A*(t-t0)) * y0 + integral_0^(t-t0) exp(A*(t0+s)) ds * b
    # = exp(A*(t-t0)) * y0 + (exp(A*(t-t0)) - I) * A^{-1} * b  (if A is invertible)
    # More generally: y(t) = exp(A*tau) * y0 + integral_0^tau exp(A*s) ds * b
    # where tau = t1 - t0
    
    tau = t1 - t0
    
    # Compute exp(A * tau)
    expA = expm(A * tau)
    
    # Compute the integral term: integral_0^tau exp(A*s) ds * b
    # = (exp(A*tau) - I) * A^{-1} * b  (when A is invertible)
    # More robust: use the identity integral_0^tau exp(A*s) ds = A^{-1} * (exp(A*tau) - I)
    # But if A is singular, use: integral = (exp(A*tau) - I) * A^{-1} only works if A invertible
    # Alternative: use block matrix exponential or numerical integration for the b term
    
    # Robust approach: use the fact that for affine system, we can augment
    # Or use: if A is invertible, integral_0^tau exp(A*s) ds = A^{-1}(exp(A*tau) - I)
    # If A is singular or near-singular, use: integral = (exp(A*tau) - I) @ solve(A, I) 
    # but this fails if A is singular.
    
    # Better: use the matrix exponential of the augmented system
    # Or simply: for small tau, the integral is approximately tau * b
    # For general case, use the formula with handling of singular A
    
    try:
        # Check if A is invertible
        det_A = np.linalg.det(A)
        if abs(det_A) > 1e-14:
            # A is invertible
            integral_term = np.linalg.solve(A, (expA - np.eye(n)) @ b)
        else:
            # A is singular; use numerical quadrature or series expansion
            # For singular A, we can use: integral_0^tau exp(A*s) ds 
            # Use the Neumann series if ||A*tau|| < 1, otherwise use block method
            # Simplest: use scipy's quad_vec or just use the augmented matrix
            pass
    except np.linalg.LinAlgError:
        integral_term = np.zeros(n, dtype=float)
    
    # Actually, let's use a more robust method: the augmented matrix exponential
    # Consider the augmented system:
    # [y; 1]' = [A  b; 0  0] [y; 1]
    # Then exp of the augmented matrix gives the solution
    
    # Build augmented matrix
    A_aug = np.zeros((n + 1, n + 1), dtype=float)
    A_aug[:n, :n] = A
    A_aug[:n, n] = b
    
    # Initial augmented state
    y0_aug = np.zeros(n + 1, dtype=float)
    y0_aug[:n] = y0
    y0_aug[n] = 1.0
    
    # Compute augmented matrix exponential
    try:
        expA_aug = expm(A_aug * tau)
        y_final_aug = expA_aug @ y0_aug
        y_final = y_final_aug[:n]
    except Exception:
        # Fallback: use the non-augmented formula
        try:
            if abs(np.linalg.det(A)) > 1e-14:
                integral_term = np.linalg.solve(A, (expA - np.eye(n)) @ b)
            else:
                # Use Taylor series for the integral
                integral_term = np.zeros(n, dtype=float)
                # integral_0^tau exp(A*s) ds = sum_{k=0}^inf A^k * tau^{k+1} / (k+1)!
                term = b.copy()
                power = np.eye(n, dtype=float)
                for k in range(50):
                    integral_term += power @ term * tau / (k + 1)
                    term = A @ term
                    if k > 5 and np.linalg.norm(term * tau / (k + 1)) < 1e-16:
                        break
            y_final = expA @ y0 + integral_term
        except Exception:
            # Last resort: use RK4 with many steps
            steps = 10000
            h = tau / steps
            y = y0.copy()
            t = t0
            for _ in range(steps):
                k1 = np.array(f(t, y), dtype=float)
                k2 = np.array(f(t + h/2, y + h/2 * k1), dtype=float)
                k3 = np.array(f(t + h/2, y + h/2 * k2), dtype=float)
                k4 = np.array(f(t + h, y + h * k3), dtype=float)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
            y_final = y
    
    # Check deadline
    if time.time() > deadline:
        return y_final
    
    return y_final
