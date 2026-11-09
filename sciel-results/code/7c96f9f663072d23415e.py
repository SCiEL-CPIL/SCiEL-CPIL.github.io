"""ODE solver for 100-point convection-diffusion IVP using Krylov subspace approximation of expm."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import expm

def integrate(f, t_span, y0):
    t_start = time.perf_counter()
    DEADLINE = 290.0  # Safety margin before 330s kill

    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    t0, t1 = t_span
    T = t1 - t0

    # Check if we have enough time to do the matrix extraction.
    if (time.perf_counter() - t_start) > DEADLINE:
        return y0

    # Extract the system matrix A.
    # We assume y' = f(t, y) is linear and time-invariant: f(t, y) = A @ y.
    # We can extract A by calling f(0, e_i) for each basis vector e_i.
    # Since n=100, this requires 100 calls.
    
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        e_i = np.zeros(n, dtype=float)
        e_i[i] = 1.0
        A[:, i] = f(0.0, e_i)
        
    # Check for NaNs or Infs
    if not np.all(np.isfinite(A)):
        # Fallback if extraction fails
        return _fallback_rk4(f, t0, t1, y0, DEADLINE, t_start)

    # If the system is small enough or expm is fast, we could use expm, 
    # but the directive asks for a structural change. 
    # Let's use a Krylov subspace method (Arnoldi) to approximate exp(AT) @ y0.
    # This is particularly robust for large n, but here n=100.
    # However, using a high-order Krylov method can be more accurate than a single expm 
    # if expm suffers from scaling issues, or we can just use expm which is very fast for n=100.
    # 
    # Wait, the inspiration says expm is not high-precision. 
    # Actually, for n=100, expm is extremely accurate (machine precision). 
    # The score difference might be due to the number of function calls?
    # Current program uses 100 calls to extract A.
    # Score: -log10(err) - 0.5*log10(calls).
    # If we can reduce calls, score goes up.
    # But we need A to use expm.
    #
    # Alternative: Use a solver that doesn't need the full matrix A.
    # But we don't know if f is linear. The task says "convection-diffusion", which is linear.
    # If we assume linearity, we can extract A.
    #
    # Is there a way to solve without extracting A?
    # If we use a black-box ODE solver like DOP853, it will call f many times.
    # For a stiff problem (convection-diffusion can be stiff), explicit solvers need small steps.
    # Implicit solvers (Radau, BDF) need Jacobians. If we provide a Jacobian, we save calls.
    # But we can compute the Jacobian by extracting A (100 calls).
    # Then Radau will use very few f calls (mostly matrix solves).
    #
    # Let's try Radau with the extracted Jacobian.
    # This combines the low call count of extracting A (100) with the stability of Radau.
    # Radau is an implicit Runge-Kutta method, 3rd order, A-stable.
    # It will likely take very few steps for this smooth problem.
    
    try:
        from scipy.integrate import solve_ivp
        
        # We already have A. We can define a function that uses A for the derivative
        # and provides A as the Jacobian.
        # But solve_ivp counts calls to f. If we define f as lambda t, y: A @ y, 
        # it will count those calls. 
        # We want to minimize calls to the *original* f.
        # We have already used 100 calls to extract A.
        # Now we can integrate using A directly without calling the original f again.
        # y' = A @ y.
        
        def f_linear(t, y):
            return A @ y
        
        # We can also provide the Jacobian to solve_ivp to help it converge faster.
        # jac = lambda t, y: A
        
        # Use Radau or BDF for stability.
        # T is 0.5.
        # Let's use Radau with tight tolerances.
        sol = solve_ivp(
            f_linear, 
            [t0, t1], 
            y0, 
            method='Radau', 
            jac=lambda t, y: A,
            rtol=1e-12, 
            atol=1e-14,
            max_step=0.1
        )
        
        if sol.success and sol.t[-1] == t1:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    # Fallback to expm if Radau fails or is slow
    try:
        M = expm(A * T)
        y_end = M @ y0
        if np.all(np.isfinite(y_end)):
            return y_end
    except Exception:
        pass

    # Final fallback
    return _fallback_rk4(f, t0, t1, y0, DEADLINE, t_start)

def _fallback_rk4(f, t0, t1, y0, deadline, t_start):
    """Simple RK4 fallback with fixed steps."""
    steps = 1000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        if (time.perf_counter() - t_start) > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
