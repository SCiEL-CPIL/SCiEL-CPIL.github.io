"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Solve the Van der Pol oscillator (mu=5) using spectral Chebyshev collocation.
    
    The method discretizes the ODE y' = f(t, y) on a grid of Chebyshev nodes,
    converting the IVP into a system of nonlinear algebraic equations which is
    solved via Newton-Raphson iteration.
    
    Parameters:
    -----------
    f : callable
        Function f(t, y) returning the derivative y'.
    t_span : tuple
        (t_start, t_end)
    y0 : array_like
        Initial condition y(t_start).
        
    Returns:
    --------
    y_end : ndarray
        The state at t_end.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n_dims = len(y0)
    
    # Number of Chebyshev nodes. 
    # For a stiff problem like Van der Pol with mu=5, we need enough resolution.
    # The directive suggests ~200 nodes. Let's start with 200.
    N = 200
    
    # Chebyshev nodes of the second kind on [t0, t1]
    # x_k = cos(k*pi/N), k=0..N
    # Map to [t0, t1]
    k = np.arange(N + 1)
    x = np.cos(k * np.pi / N)
    # Map from [-1, 1] to [t0, t1]
    # t = 0.5 * (t1 - t0) * x + 0.5 * (t1 + t0)
    t = 0.5 * (t1 - t0) * x + 0.5 * (t1 + t0)
    
    # We solve for y at all N+1 nodes.
    # Let Y be a matrix of shape (N+1, n_dims) where Y[i] = y(t_i).
    # The collocation condition is:
    # D @ Y = f(t, Y)
    # where D is the Chebyshev differentiation matrix.
    
    # Construct the Chebyshev differentiation matrix D (size (N+1)x(N+1))
    # Standard formula for Chebyshev points of the second kind.
    c = np.ones(N + 1)
    c[0] = 2
    c[-1] = 2
    c[1:-1] *= -1
    c *= (-1) ** k
    
    # x_k = cos(k*pi/N)
    # D_{ij} = (x_i - x_j) / (x_i^2 - 1) * (c_i / c_j) for i != j
    # D_{ii} = -x_i / (2 * (x_i^2 - 1)) for i != 0, N
    # D_{00} = (2*N^2 + 1) / 6
    # D_{NN} = -(2*N^2 + 1) / 6
    
    D = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            if i == j:
                if i == 0:
                    D[i, j] = (2 * N ** 2 + 1) / 6.0
                elif i == N:
                    D[i, j] = -(2 * N ** 2 + 1) / 6.0
                else:
                    D[i, j] = -x[i] / (2 * (x[i] ** 2 - 1))
            else:
                D[i, j] = (c[i] / c[j]) * 1.0 / (x[i] - x[j])
    
    # Scale D to map from [-1, 1] to [t0, t1]
    # If t = a*x + b, then dt/dx = a.
    # y'(t) = dy/dt = dy/dx * dx/dt = dy/dx * (1/a)
    # So D_t = D / a
    a = 0.5 * (t1 - t0)
    D = D / a
    
    # We have the system: D @ Y = F(Y)
    # where F(Y) is a vector of f(t_i, Y[i]) for all i.
    # We also have the initial condition: Y[0] = y0.
    
    # Let's use Newton-Raphson on the full system.
    # Unknowns: Y (shape (N+1, n_dims)).
    # Equations:
    # 1. Y[0] - y0 = 0  (n_dims equations)
    # 2. D @ Y - F(Y) = 0  ((N+1)*n_dims equations)
    # Total unknowns: (N+1)*n_dims
    # Total equations: n_dims + (N+1)*n_dims = (N+2)*n_dims
    # This is an overdetermined system. We should drop the equation at t0 from the collocation
    # or treat the initial condition as a constraint and solve for the rest.
    
    # Better approach:
    # Unknowns: Y[1:] (shape (N, n_dims))
    # Equations: D @ Y - F(Y) = 0 for all i=0..N
    # But D @ Y depends on Y[0] which is fixed.
    # Let Y_full = [y0; Y_unknowns]
    # Residual R = D @ Y_full - F(Y_full)
    # We solve R = 0 for Y_unknowns.
    
    # Initial guess: linear interpolation or just constant y0?
    # For Van der Pol, it oscillates. A constant guess might be far.
    # Let's use a coarse RK4 solution as initial guess to speed up convergence.
    # Or just use y0 for all nodes.
    
    Y_full = np.tile(y0, (N + 1, 1))
    
    # Newton-Raphson iterations
    max_iter = 50
    tol = 1e-14
    
    for _ in range(max_iter):
        # Evaluate F(Y_full)
        F = np.empty((N + 1, n_dims))
        for i in range(N + 1):
            F[i] = f(t[i], Y_full[i])
        
        # Residual
        R = D @ Y_full - F
        
        # We only update Y_full[1:]
        # J is the Jacobian of R with respect to Y_full[1:]
        # R = D @ Y_full - F(Y_full)
        # dR/dY_j = D[:, j] - dF/dY_j
        # Since F depends on Y element-wise (f(t_i, Y_i) only depends on Y_i),
        # dF/dY is block diagonal.
        
        # We need to solve J @ delta_Y = -R[1:]
        # where J is of shape ((N+1)*n_dims, N*n_dims)
        
        # Let's build J block by block.
        # J[i, j] corresponds to dR_i / dY_j
        # If i == j, J[i, j] = D[i, j] * I - J_f(t_i, Y_i)
        # If i != j, J[i, j] = D[i, j] * I
        
        # This is a large sparse matrix, but for N=200, it's 400x400 (if n_dims=2).
        # We can use dense linear algebra.
        
        size = N * n_dims
        J = np.zeros(((N + 1) * n_dims, size))
        
        # Precompute Jacobians of f at each node?
        # f is a black box. We must use finite differences.
        # This will cost (N+1) * n_dims * 2 calls per Newton iteration.
        # With N=200, n_dims=2, that's 800 calls per iteration.
        # 50 iterations -> 40,000 calls.
        # Limit is 2,000,000. This is safe.
        
        eps = 1e-8
        
        for j in range(N):  # j corresponds to Y_full[j+1]
            for dim in range(n_dims):
                col_idx = j * n_dims + dim
                # Perturb Y_full[j+1, dim]
                Y_pert = Y_full.copy()
                Y_pert[j + 1, dim] += eps
                
                F_pert = np.empty((N + 1, n_dims))
                for i in range(N + 1):
                    F_pert[i] = f(t[i], Y_pert[i])
                
                R_pert = D @ Y_pert - F_pert
                
                # Column of J
                J[:, col_idx] = (R_pert - R) / eps
        
        # Solve J @ delta = -R[1:]
        # R[1:] is the residual for rows 1..N
        R_reduced = R[1:].flatten()
        
        try:
            delta = np.linalg.lstsq(J, -R_reduced, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        
        delta_Y = delta.reshape((N, n_dims))
        
        Y_full[1:] += delta_Y
        
        # Check convergence
        if np.linalg.norm(delta_Y) < tol * (1 + np.linalg.norm(Y_full[1:])):
            break
    
    return Y_full[-1]
# EVOLVE-BLOCK-END
