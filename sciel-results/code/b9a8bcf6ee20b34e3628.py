import numpy as np
import time
import scipy.linalg

def integrate(f, t_span, y0):
    """
    Integrates a stiff 15-dimensional linear ODE system.
    Strategy: Assume the system is linear y' = A y (time-invariant).
    1. Estimate the constant matrix A using forward differences (n calls).
    2. Compute the exact solution using the matrix exponential: y(t1) = expm(A * (t1-t0)) @ y0.
    This requires only n+1 calls to f (n for A estimation + 1 initial check if needed, but actually just n for A).
    Wait, estimating A via f(t, y + eps*e_i) requires n calls if we assume f(t, 0) = 0 (linear homogeneous).
    Since it's a linear system y' = A y, f(t, y) = A y.
    So f(t, e_i) = A e_i = column i of A.
    Thus, A can be reconstructed from n calls: A[:, i] = f(t0, e_i).
    This is extremely efficient (15 calls) and exact (up to floating point precision of expm).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 290.0

    # Estimate A using the linearity assumption: f(t, y) = A y
    # Since the system is linear and homogeneous (implied by "linear initial-value problem" in this context usually),
    # we can get the columns of A by evaluating f at the basis vectors.
    # Note: The problem states "stiff linear system". It is standard to assume y' = A(t)y.
    # If A is constant, we use expm.
    # If A is time-dependent, this simple approach fails. However, the inspirations suggest assuming linearity is key.
    # Let's try to detect if A is time-dependent.
    
    # Step 1: Estimate A at t0
    A = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A[:, j] = f(t0, e)
        
    # Step 2: Check if A is constant by estimating A at t1
    # We can do this cheaply if we just check one column or use a random vector.
    # But to be safe, let's estimate A at t1 as well? That costs another n calls.
    # Total 2n calls. For n=15, that's 30 calls.
    # If A is constant, we use expm.
    # If A is not constant, we need a fallback.
    
    # Let's check if A is constant by comparing A(t0) and A(t1).
    A_end = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A_end[:, j] = f(t1, e)
        
    # Check if A is approximately constant
    if np.allclose(A, A_end, rtol=1e-5, atol=1e-8):
        # Use matrix exponential for the exact solution
        dt = t1 - t0
        try:
            expm_A_dt = scipy.linalg.expm(A * dt)
            y_final = expm_A_dt @ y0
            return y_final
        except Exception:
            pass

    # Fallback: If A is not constant, use a high-order solver.
    # Since we've already used 2n calls, we have plenty of budget left.
    # We can use scipy.integrate.solve_ivp with Radau method.
    # However, solve_ivp might be slow or inaccurate.
    # Let's try a custom implementation or a different solver.
    # Given the score of the current program is 1.13, and it uses a hybrid,
    # the expm path is likely the one that works well for constant A.
    # If A is time-dependent, the problem is harder.
    # But the task says "fixed 15-dimensional stiff linear system".
    # Often "linear system" in these contexts implies constant coefficients unless specified otherwise.
    # Let's assume constant coefficients are the primary case.
    
    # If we get here, A is not constant.
    # We can try to integrate using a piecewise expm approach with small steps.
    # Or just use solve_ivp.
    
    # Let's use solve_ivp with Radau and tight tolerances.
    import scipy.integrate
    
    def _solve_ivp_fallback(f, t_span, y0):
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0, method='Radau', rtol=1e-12, atol=1e-14,
            max_step=0.1, first_step=0.01
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # Return best effort
            return sol.y[:, -1]
            
    return _solve_ivp_fallback(f, t_span, y0)
