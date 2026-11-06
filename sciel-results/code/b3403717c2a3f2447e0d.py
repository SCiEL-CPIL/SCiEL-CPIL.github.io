"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate
import scipy.linalg

def integrate(f, t_span, y0):
    """
    Main integration function exploiting the linear structure of the 
    convection-diffusion ODE.
    
    Strategy:
    1. The ODE is y' = A @ y (linear, time-independent).
    2. We determine A by evaluating f at a few points. Since f(t, y) is linear in y,
       we can use f(0, e_i) to find columns of A, or more efficiently, 
       if we assume the standard form, we can reconstruct A.
       
       However, a more robust way for a generic linear f(t,y) = A@y is:
       A = f(0, y0) / y0 is not element-wise correct if A is a matrix.
       
       Better approach:
       Since the problem is fixed 100-point convection-diffusion, A is a sparse tridiagonal matrix.
       We can reconstruct A by calling f(0, v) for basis vectors? That's 100 calls.
       Or, we can use the fact that for linear systems, we can estimate A using:
       A ≈ f(0, y0) is not enough to get the whole matrix.
       
       Wait, the prompt says "fixed 100-point convection-diffusion". 
       Usually, this implies y' = D @ y + c @ y (diffusion + convection).
       
       Let's try a different approach:
       If we call f(0, y0), we get y'_0 = A @ y0.
       If we call f(0, 2*y0), we get y'_1 = A @ (2*y0) = 2 * y'_0. This confirms linearity but doesn't give A.
       
       To get A, we need to probe the space.
       However, we can use the property that for a linear system, the solution is exp(A*t) @ y0.
       We can approximate A by using a finite difference in time? No, that uses f calls.
       
       Actually, there is a trick: if we can call f with arbitrary y, we can determine A by calling
       f(0, e_i) for i=0..N-1. That is N calls (100 calls).
       Then we compute expm(A*t) @ y0.
       Total calls: 100.
       Accuracy: Exact (up to floating point and expm precision).
       
       Is 100 calls better than ~1000+ calls?
       Score = -log10(err) - 0.5 * log10(calls)
       If err is ~1e-15 (from expm), -log10(err) ~ 15.
       If calls = 100, -0.5 * log10(100) = -1.
       Score ~ 14.
       
       Current program:
       Calls ~ 1000-5000.
       -0.5 * log10(1000) = -1.5.
       -0.5 * log10(5000) = -1.9.
       Error is controlled by rtol. With rtol=1e-13, err ~ 1e-13.
       -log10(err) ~ 13.
       Score ~ 11.5.
       
       So 100 calls with expm should be better.
       
       Can we do fewer than 100 calls?
       If we know the structure (tridiagonal), we only need to determine the diagonals.
       But we don't know the structure for sure without inspecting the problem.
       However, "fixed 100-point convection-diffusion" strongly implies a specific matrix structure.
       
       Let's stick to the general linear reconstruction:
       1. Call f(0, e_i) for i in range(N). This gives column i of A.
       2. Construct A.
       3. Compute expm(A * (t1 - t0)) @ y0.
       
       Cost: N calls to f.
       For N=100, this is 100 calls.
       
       Optimization:
       We can check if the system is linear.
       y0 is given.
       If we just want to solve for this specific y0, do we need the full A?
       Yes, because exp(A*t) depends on the full matrix.
       
       Alternative:
       Use the scaling and squaring method for matrix exponential, which is what expm does.
       
       Let's implement the 100-call strategy.
       
       Edge case: What if f is not linear? The problem states it is convection-diffusion, which is linear.
       What if t_span is not starting at 0? f(t, y). For linear time-independent systems, f(t, y) = A @ y.
       The time t does not appear in the derivative for this specific problem class (autonomous).
       Even if it's non-autonomous, convection-diffusion with constant coefficients is autonomous.
       
       Let's verify linearity quickly with 2 calls?
       f(0, y0) and f(0, 2*y0). If f(0, 2*y0) == 2 * f(0, y0), it's likely linear.
       Then proceed to get columns.
       
       To be safe and fast:
       1. Get y0.
       2. Create identity matrix I.
       3. A = np.zeros((N, N))
       4. For i in range(N):
            A[:, i] = f(0, I[:, i])
       5. M = scipy.linalg.expm(A * (t1 - t0))
       6. return M @ y0
       
       This uses exactly N calls to f.
       
       Time complexity:
       N calls to f (fast, just matrix-vector mults inside f).
       expm is O(N^3). For N=100, this is 1,000,000 operations, which is very fast (< 1ms).
       
       This should be extremely fast and accurate.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    start_time = time.time()
    deadline = start_time + 295.0

    # Check if we have time
    if time.time() > deadline - 5.0:
        return y0  # Fallback, shouldn't happen

    # Reconstruct A
    # We assume f(t, y) is linear in y and independent of t (autonomous)
    # This is true for constant coefficient convection-diffusion
    A = np.zeros((N, N), dtype=float)
    
    # Use the first few columns to check linearity/autonomy?
    # Let's just get the columns.
    # To save time, we can parallelize? No, f is a Python function call, likely overhead bound.
    # But 100 calls is very small.
    
    for i in range(N):
        e_i = np.zeros(N)
        e_i[i] = 1.0
        # Evaluate at t0 to be safe, though t shouldn't matter for autonomous
        A[:, i] = f(t0, e_i)
        
        # Safety check: if time is running out, break and fallback?
        # 100 calls should take < 1ms, so this is safe.
        
    # Compute matrix exponential
    # dt = t1 - t0
    dt = t1 - t0
    
    # If dt is 0, return y0
    if dt == 0:
        return y0

    try:
        M = scipy.linalg.expm(A * dt)
        y_final = M @ y0
        return y_final
    except Exception:
        # Fallback to standard solver if expm fails (e.g., NaNs)
        result = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15
        )
        if result.success and result.t[-1] == t1:
            return result.y[:, -1]
        return y0

# EVOLVE-BLOCK-END
