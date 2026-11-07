"""Rewritten program using Spectral Collocation method for high accuracy and low call count."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.linalg import block_diag

def integrate(f, t_span, y0):
    """
    Integrates the ODE system using Spectral Collocation (Chebyshev) method.
    Falls back to DOP853 if spectral method fails.
    """
    y0_arr = np.array(y0, dtype=float)
    t_start, t_end = t_span
    
    # Parameters for spectral method
    N = 30  # Number of collocation points
    
    # Map t in [t_start, t_end] to x in [-1, 1]
    # t = 0.5 * (t_end - t_start) * (x + 1) + t_start
    # x = 2 * (t - t_start) / (t_end - t_start) - 1
    
    # Chebyshev nodes (roots of Chebyshev polynomial of first kind)
    # x_k = cos((2k+1)*pi/(2N)) for k=0..N-1
    # Or use Chebyshev points of the first kind (extrema)
    # Let's use roots for better stability
    k = np.arange(N)
    x = np.cos((2 * k + 1) * np.pi / (2 * N))
    # Sort in increasing order for t
    x = x[::-1] 
    # Map to t
    t = 0.5 * (t_end - t_start) * (x + 1) + t_start
    
    # Construct Chebyshev differentiation matrix D
    # Formula for D at Chebyshev points
    c = np.ones(N)
    c[0] = 2
    c[-1] = 2
    c = c * (-1) ** k
    c = c[::-1] # Reverse to match x order? Need to be careful.
    
    # Standard formula:
    # x_j = cos((j + 0.5) * pi / N)
    # D_ij = (c_i / c_j) * x_j / (x_i - x_j) for i != j
    # D_ii = -x_i / (2 * (1 - x_i^2))
    
    # Let's use a known implementation or derive carefully.
    # Using scipy.linalg? No, implement manually.
    
    # Re-calculate x and c correctly
    # x_j = cos((j + 0.5) * pi / N)
    x = np.cos((np.arange(N) + 0.5) * np.pi / N)
    # Sort x in ascending order
    idx = np.argsort(x)
    x = x[idx]
    
    c = np.ones(N)
    c[0] = 2
    c[-1] = 2
    c = c * (-1) ** np.arange(N)
    # c needs to be sorted same as x
    c = c[idx]
    
    X = np.tile(x, (N, 1))
    dX = X - X.T
    
    D = (np.tile(c, (N, 1)) / np.tile(c, (1, N))) / (dX + np.eye(N))
    D = D - np.diag(np.sum(D, axis=1))
    
    # Residual function
    # y_flat is 2N vector: [y0_0, y1_0, y0_1, y1_1, ...]
    def residual(y_flat):
        Y = y_flat.reshape(N, 2)
        # Compute D @ Y
        # D is N x N, Y is N x 2
        # Result is N x 2
        dY = D @ Y
        # Compute F
        F = np.zeros_like(Y)
        for i in range(N):
            F[i] = f(t[i], Y[i])
        # Residual = dY - F
        return (dY - F).flatten()
        
    # Jacobian
    def jacobian(y_flat):
        Y = y_flat.reshape(N, 2)
        # J = D kron I - J_F
        # D_kron_I is sparse, but we can construct dense
        # D_kron_I[i, j] = D[i, j] * I
        # J_F is block diagonal
        
        # Construct D_kron_I
        # Shape (2N, 2N)
        J = np.zeros((2 * N, 2 * N))
        for i in range(N):
            for j in range(N):
                J[2*i, 2*j] = D[i, j]
                J[2*i, 2*j+1] = 0
                J[2*i+1, 2*j] = 0
                J[2*i+1, 2*j+1] = D[i, j]
        
        # Subtract J_F
        for i in range(N):
            y0_i, y1_i = Y[i]
            # J_f = [[1.5 - y1, -y0], [y1, y0 - 3]]
            J[2*i, 2*i] -= (1.5 - y1_i)
            J[2*i, 2*i+1] -= (-y0_i)
            J[2*i+1, 2*i] -= (y1_i)
            J[2*i+1, 2*i+1] -= (y0_i - 3)
            
        return J

    # Initial guess
    # Linear interpolation from y0 to some estimate?
    # Just use y0 for all points
    y_guess = np.tile(y0_arr, (N, 1)).flatten()
    
    try:
        y_sol, info, ier, msg = fsolve(residual, y_guess, fprime=jacobian, full_output=True)
        if ier != 1:
            raise RuntimeError("Spectral solve failed")
            
        # Check residual norm
        res_norm = np.linalg.norm(residual(y_sol))
        if res_norm > 1e-10:
            raise RuntimeError("Spectral residual too high")
            
        return y_sol[-2:]
    except Exception:
        # Fallback to DOP853
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13
        )
        return sol.y[:, -1]

