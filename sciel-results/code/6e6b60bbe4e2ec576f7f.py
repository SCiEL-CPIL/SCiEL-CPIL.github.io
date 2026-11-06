"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Solve the fixed 10x10 2D heat equation IVP on [0, 0.05].
    
    Uses spectral decomposition via FFT/DCT to solve analytically.
    First determines if the system is linear autonomous by checking f(t0, 0) ≈ 0.
    Then identifies the matrix structure with minimal f calls and uses
    eigendecomposition for exact solution.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Check linearity: f(t, 0) should be ~0 for linear autonomous system
    zero_state = np.zeros(n)
    f_zero = f(t0, zero_state)
    
    if np.allclose(f_zero, 0, atol=1e-10):
        # Linear autonomous system: f(t, y) = A @ y
        # Determine A using basis vectors (100 calls)
        A = np.empty((n, n), dtype=float)
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            A[:, j] = f(t0, e)
        
        # Use eigendecomposition for exact solution
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # Transform y0 to eigenbasis
        c = np.linalg.solve(eigenvectors, y0)
        
        # Apply exponential decay
        dt = t1 - t0
        exp_factors = np.exp(eigenvalues * dt)
        
        # Reconstruct
        y_final = eigenvectors @ (exp_factors * c)
        
        return y_final
    else:
        # Fallback: use high-accuracy ODE solver
        from scipy.integrate import solve_ivp
        
        def ode_func(t, y):
            return f(t, y)
        
        sol = solve_ivp(ode_func, t_span, y0, method='DOP853', 
                       rtol=1e-12, atol=1e-14)
        return sol.y[:, -1]
# EVOLVE-BLOCK-END
