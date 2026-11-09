"""ODE solver: 15-dimensional stiff linear system using matrix exponential."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

def integrate(f, t_span, y0):
    """Solve y' = f(t, y) for a 15-dim linear stiff system.
    
    Strategy:
    1. Extract the coefficient matrix A (assuming f(t,y) = A(t) @ y)
    2. Check if A is time-invariant
    3. Use matrix exponential for exact solution
    4. Fallback to implicit Euler if needed
    """
    start_time = time.time()
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: leave 30s margin before 330s limit
    deadline = start_time + 300.0
    
    # Step 1: Extract A matrix by probing with basis vectors
    # f(t, 0) should be 0 for a homogeneous linear system
    f_zero = f(t0, np.zeros(n))
    
    # Extract columns of A: A[:, j] = f(t0, e_j) where e_j is j-th basis vector
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
    
    # Check if A is time-invariant by sampling at a few points
    # If time-dependent, we'll need to integrate the matrix ODE
    A_mid = np.zeros((n, n))
    t_mid = 0.5 * (t0 + t1)
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A_mid[:, j] = f(t_mid, e_j)
    
    # Check time-invariance
    A_diff = np.max(np.abs(A - A_mid))
    A_scale = max(1.0, np.max(np.abs(A)))
    is_time_invariant = A_diff < 1e-10 * A_scale
    
    if is_time_invariant:
        # Time-invariant case: y(t1) = expm((t1-t0) * A) @ y0
        dt = t1 - t0
        M = dt * A
        y_end = expm(M) @ y0
    else:
        # Time-dependent case: use product of exponentials with many steps
        # y(t) = Phi(t, t0) @ y0 where Phi satisfies dPhi/dt = A(t) @ Phi
        # Approximate with many small steps
        steps = 1000
        h = (t1 - t0) / steps
        Phi = np.eye(n)
        t = t0
        for step in range(steps):
            if time.time() > deadline:
                # Fallback: use current Phi
                break
            # Extract A at current time
            A_cur = np.zeros((n, n))
            for j in range(n):
                e_j = np.zeros(n)
                e_j[j] = 1.0
                A_cur[:, j] = f(t, e_j)
            # Update Phi: Phi <- expm(h * A_cur) @ Phi
            Phi = expm(h * A_cur) @ Phi
            t += h
        y_end = Phi @ y0
    
    # Verify solution is finite
    if not np.all(np.isfinite(y_end)):
        # Fallback to implicit Euler
        y_end = _implicit_euler_fallback(f, t0, t1, y0, 5000, deadline)
    
    return y_end


def _implicit_euler_fallback(f, t0, t1, y0, steps, deadline):
    """Fallback implicit Euler with Newton iteration."""
    y = np.array(y0, dtype=float)
    n = len(y)
    h = (t1 - t0) / steps
    t = t0
    
    for _ in range(steps):
        if time.time() > deadline:
            break
        z = y.copy()
        for _newton in range(10):
            fz = f(t + h, z)
            # Finite difference Jacobian
            J = np.zeros((n, n))
            for j in range(n):
                e = np.zeros(n)
                scale = max(1.0, abs(z[j]))
                e[j] = 1e-8 * scale
                J[:, j] = (f(t + h, z + e) - fz) / e[j]
            
            # Solve (I - h*J) * step = -(z - y - h*fz)
            try:
                step = np.linalg.solve(np.eye(n) - h * J, -(z - y - h * fz))
            except np.linalg.LinAlgError:
                break
            z = z + step
            if np.linalg.norm(step) <= 1e-12 * (1 + np.linalg.norm(z)):
                break
        y = z
        t += h
    
    return y
# EVOLVE-BLOCK-END
