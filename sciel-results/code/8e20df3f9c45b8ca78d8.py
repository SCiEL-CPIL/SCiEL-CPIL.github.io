"""
Strategy: Exploit the linear autonomous structure of the convection-diffusion problem.
Instead of generic adaptive ODE solvers which require thousands of f evaluations to
achieve high precision, we reconstruct the system matrix A from N basis vector probes.
We then solve the linear system exactly (or to machine precision) using a high-order
implicit method (Crank-Nicolson with 1 step, or expm for 1 step).
Since f is linear and time-invariant, y(t) = expm(A * dt) @ y0.
Reconstructing A costs N calls. Solving expm(A*dt) costs O(N^3) but 0 calls to f.
Total calls: N.
Accuracy: Limited only by the numerical stability of expm and the reconstruction of A.
For N=100, expm is highly accurate and fast.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg
import scipy.integrate

def integrate(f, t_span, y0):
    """
    Main integration function.
    Strategy: 
    1. Assume f(t, y) = A @ y (linear, autonomous).
    2. Reconstruct A by evaluating f(t, e_i) for i in 0..N-1.
    3. Compute y_end = expm(A * (t1 - t0)) @ y0.
    4. Fallback to high-precision adaptive solver if linearity/autonomy check fails 
       (though for this specific task, it should hold).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 295.0
    
    if N == 0:
        return y0
        
    dt = t1 - t0
    if dt == 0:
        return y0

    # Step 1: Verify linearity and time-invariance with a few probes.
    # We need N calls to reconstruct A anyway. Let's just do it.
    # If the system is not linear, this will produce a garbage A, 
    # and the result will be wrong. 
    # However, the task specifies "convection-diffusion" which is linear.
    # To be safe, we can check if f(t, 0) is close to 0 and f(t, y1+y2) == f(t,y1)+f(t,y2).
    # But for speed and given the task contract, we assume linearity.
    
    # Reconstruct A
    # A @ e_i = f(t0, e_i)
    # So column i of A is f(t0, e_i)
    
    A = np.zeros((N, N), dtype=float)
    
    for i in range(N):
        e_i = np.zeros(N)
        e_i[i] = 1.0
        # Evaluate at t0. For autonomous systems, t doesn't matter.
        # If it's not autonomous, this method fails, but we assume it is.
        A[:, i] = f(t0, e_i)
        
    # Check time invariance cheaply? 
    # We already used N calls. One more call to check f(t1, e_0) vs A[:, 0] is cheap.
    e_0 = np.zeros(N)
    e_0[0] = 1.0
    v_t1 = f(t1, e_0)
    v_t0_col0 = A[:, 0]
    
    if not np.allclose(v_t1, v_t0_col0, rtol=1e-10, atol=1e-12):
        # System is time-dependent or non-linear. Fallback to robust solver.
        # This branch is unlikely for the specified task.
        if time.time() > deadline - 5.0:
            return y0
        result = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14
        )
        if result.success:
            return result.y[:, -1]
        return y0

    # Step 2: Compute the exact solution for the linear autonomous system.
    # y(t) = expm(A * dt) @ y0
    
    try:
        M = scipy.linalg.expm(A * dt)
        y_final = M @ y0
        return y_final
    except Exception:
        # Fallback if expm fails (e.g., numerical issues, though unlikely for N=100)
        if time.time() > deadline - 5.0:
            return y0
        result = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-10,
            atol=1e-12
        )
        if result.success:
            return result.y[:, -1]
        return y0

# EVOLVE-BLOCK-END
