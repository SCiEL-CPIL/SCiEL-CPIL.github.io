"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _extract_matrix(f, t0, y0):
    """
    Extract the system matrix A for a linear ODE y' = A y.
    Uses n function calls.
    Strategy: Use y0 as the first basis vector and standard unit vectors e_0 ... e_{n-2} for the rest.
    If the resulting matrix is ill-conditioned, fall back to the standard basis e_0 ... e_{n-1}.
    """
    n = len(y0)
    
    # Construct the basis matrix Y
    # Columns: y0, e0, e1, ..., e_{n-2}
    Y = np.zeros((n, n))
    Y[:, 0] = y0
    for j in range(1, n):
        Y[j-1, j] = 1.0
        
    # Check condition number to ensure stability
    use_fallback = False
    try:
        cond = np.linalg.cond(Y)
        if cond > 1e10:
            use_fallback = True
    except np.linalg.LinAlgError:
        use_fallback = True
        
    if use_fallback:
        Y = np.eye(n)
        
    # Extract A by evaluating f at the basis vectors
    F = np.empty((n, n))
    for j in range(n):
        F[:, j] = f(t0, Y[:, j])
        
    # Solve F = A Y  =>  A = F Y^{-1}
    # Using solve is more stable than explicit inverse
    try:
        A = np.linalg.solve(Y.T, F.T).T
    except np.linalg.LinAlgError:
        # This should not happen if Y is well-conditioned, but just in case
        # If it fails, we assume Y was bad and we should have fallen back.
        # If we are here, it's a critical error.
        # We can try to recover by using the standard basis if we didn't already.
        if not use_fallback:
             Y = np.eye(n)
             F = np.empty((n, n))
             for j in range(n):
                 F[:, j] = f(t0, Y[:, j])
             A = np.linalg.solve(Y.T, F.T).T
        else:
             # If even the identity fails, something is very wrong.
             # But identity always works.
             A = np.linalg.solve(Y.T, F.T).T
             
    return A


def _solve_linear_ode_explicit(A, t0, t1, y0):
    """
    Solve y' = A y using matrix exponential via scipy.
    """
    from scipy.linalg import expm
    dt = t1 - t0
    M = expm(A * dt)
    return M @ y0


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Extract A using n calls
    A = _extract_matrix(f, t0, y0)
    
    # Solve using matrix exponential
    y_end = _solve_linear_ode_explicit(A, t0, t1, y0)
    
    return y_end
# EVOLVE-BLOCK-END
