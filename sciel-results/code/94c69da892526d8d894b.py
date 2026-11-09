"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    n = len(y0_arr)
    
    # 1. Determine if the system is linear/affine and extract A and b
    # We assume the system is y' = A(t)*y + b(t).
    # The task description says "fixed 15-dimensional stiff linear system".
    # "Fixed" usually implies constant coefficients.
    # We will probe at t0 to get A and b.
    
    # Probe b(t0) = f(t0, 0)
    y_zero = np.zeros(n)
    b_t0 = np.array(f(t0, y_zero), dtype=float)
    
    # Probe A(t0) columns
    # We use a small perturbation.
    # To be robust against scaling issues, we can use 1.0 and check consistency, 
    # or just use 1.0 since n=15 is small.
    # Let's use 1.0 for the probe vector components to avoid floating point division by tiny numbers.
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        f_j = np.array(f(t0, e_j), dtype=float)
        # f(t, y) = A y + b
        # f(t, e_j) = A e_j + b = A[:, j] + b
        # So A[:, j] = f(t, e_j) - b
        A[:, j] = f_j - b_t0
        
    # Verify linearity and time-invariance?
    # The problem states "fixed ... linear system".
    # Let's verify if f(t1, y) matches A*y + b_t0.
    # If it doesn't, the system is time-dependent or non-linear.
    # However, the score is based on a reference solution.
    # If the system is time-dependent y' = A(t)y, a single expm won't work directly 
    # without product integration.
    # But "fixed" strongly suggests constant A.
    
    # Let's check if A is constant by probing at t1 (midpoint or end)
    # Actually, let's just check if the system is linear at t1 with the same A and b.
    # If it fails, we fall back to a robust ODE solver.
    
    # Check at t1
    # But wait, if it's affine y' = A y + b, the solution is:
    # y(t) = exp(A(t-t0)) y0 + integral_{t0}^t exp(A(t-s)) b(s) ds
    # If b is constant, y(t) = exp(A(t-t0)) y0 + (exp(A(t-t0)) - I) A^{-1} b
    # Or simpler: define z = [y; 1], z' = [[A, b], [0, 0]] z.
    
    # Let's assume constant A and constant b first.
    # Verify A and b at t1
    b_t1 = np.array(f(t1, y_zero), dtype=float)
    A_t1 = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        f_j_t1 = np.array(f(t1, e_j), dtype=float)
        A_t1[:, j] = f_j_t1 - b_t1
        
    # Check if A and b are constant
    # Tolerance for "fixed"
    tol = 1e-10
    is_constant = (np.allclose(A, A_t1, atol=tol) and np.allclose(b_t0, b_t1, atol=tol))
    
    if is_constant:
        # System is y' = A y + b with constant A, b
        # Use augmented matrix method to handle the affine term exactly
        # z = [y; 1]
        # z' = [[A, b], [0, 0]] z
        # z(t) = exp(M * (t1 - t0)) z(t0)
        
        dt = t1 - t0
        if abs(dt) < 1e-15:
            return y0_arr
            
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = A
        M[:n, n] = b_t0
        # Last row is zero
        
        z0 = np.zeros(n + 1)
        z0[:n] = y0_arr
        z0[n] = 1.0
        
        # Compute matrix exponential
        # expm is robust for 16x16 matrices
        expM = expm(M * dt)
        
        z_final = expM @ z0
        y_final = z_final[:n]
        
        # Sanity check: if the result is not finite, fall back
        if not np.all(np.isfinite(y_final)):
            raise ValueError("Matrix exponential produced non-finite results")
            
        return y_final
        
    else:
        # Fallback: System is not constant linear/affine, or probes failed.
        # Use solve_ivp with BDF and high precision.
        from scipy.integrate import solve_ivp
        
        # We already spent n+1 calls.
        # Use a tight tolerance.
        sol = solve_ivp(
            f, (t0, t1), y0_arr,
            method='BDF',
            rtol=1e-13,
            atol=1e-15,
            max_step=abs(dt) / 50
        )
        
        if not sol.success:
             # Try even tighter or different method?
             # For now, return what we have or raise.
             # Given the contract, we should try to return something valid.
             # Let's try Radau which is also good for stiff.
             sol = solve_ivp(
                f, (t0, t1), y0_arr,
                method='Radau',
                rtol=1e-12,
                atol=1e-14
             )
             
        return sol.y[:, -1]
# EVOLVE-BLOCK-END
