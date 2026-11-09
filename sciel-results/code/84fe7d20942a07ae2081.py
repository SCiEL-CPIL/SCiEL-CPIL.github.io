"""Chebyshev spectral collocation for the Van der Pol oscillator."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _vdp_f(t, y):
    """Van der Pol ODE right-hand side (mu=5)."""
    y0, y1 = y[0], y[1]
    return np.array([y1, 5.0 * (1.0 - y0 * y0) * y1 - y0])


def _vdp_jac(t, y):
    """Analytic Jacobian of the Van der Pol ODE."""
    y0, y1 = y[0], y1
    return np.array([[0.0, 1.0],
                     [-10.0 * y0 * y1 - 1.0, 5.0 * (1.0 - y0 * y0)]])


def _chebyshev_nodes(N):
    """Chebyshev-Gauss-Lobatto nodes mapped to [0, 10]."""
    j = np.arange(N + 1)
    # Map from [-1, 1] to [0, 10]
    cos_vals = np.cos(np.pi * j / N)
    t_nodes = 5.0 * (1.0 - cos_vals)  # maps [-1,1] -> [10, 0], but order is fine
    # Actually let's use standard mapping: t = (t_end + t_start)/2 + (t_end - t_start)/2 * cos
    # cos(pi*j/N) goes from 1 to -1 as j goes 0 to N
    # So t = 5 + 5*cos(pi*j/N) goes from 10 to 0
    # We want t from 0 to 10, so flip: t = 5 - 5*cos(pi*j/N)
    t_nodes = 5.0 - 5.0 * np.cos(np.pi * j / N)
    return t_nodes


def _chebyshev_differential_matrix(N):
    """
    First derivative matrix for Chebyshev-Gauss-Lobatto nodes on [-1, 1].
    D[i,j] such that (D * y)[i] = dy/dx at node i.
    """
    x = np.cos(np.pi * np.arange(N + 1) / N)  # from 1 to -1
    c = np.ones(N + 1)
    c[0] = 0.5
    c[-1] = 0.5
    if N % 2 == 0:
        c[-1] = 1.0  # Actually for even N, c[-1] = 0.5 as well; let me re-check
    # Standard: c_0 = c_N = 1/2 (if N>0), else c_0=1
    c = np.ones(N + 1)
    c[0] = 0.5
    c[-1] = 0.5

    X, J = np.meshgrid(x, x, indexing='ij')
    D = (c[None, :] / c[:, None]) * (1.0 / (X - J + np.eye(N + 1) * 2.0))
    # Fix diagonal
    np.fill_diagonal(D, 0.0)
    D -= np.diag(np.sum(D, axis=1))

    # Actually the standard formula:
    # D = (c_i / c_j) * 1/(x_i - x_j) for i != j
    # D_ii = -sum_{j != i} D_ij = (N^2 + 1)/2 * (-1)^i / (1 - x_i^2) ... 
    # Let me use the simpler construction:
    # D_ij = (c_i / c_j) / (x_i - x_j) for i != j
    # D_ii = -sum_{j != i} D_ij
    D = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = (c[i] / c[j]) / (x[i] - x[j])
        D[i, i] = -np.sum(D[i, :])
    return D, x


def _chebyshev_differential_matrix_fast(N):
    """
    Fast construction of the Chebyshev differentiation matrix on [-1,1].
    Returns D (size (N+1)x(N+1)) and x (nodes from 1 to -1).
    """
    x = np.cos(np.pi * np.arange(N + 1) / N)  # x[0]=1, x[N]=-1
    c = np.ones(N + 1)
    c[0] = 0.5
    c[-1] = 0.5

    # Vectorized off-diagonal
    X = np.tile(x, (N + 1, 1))  # (N+1, N+1), X[i,j] = x[j]
    J = np.tile(x[:, None], (1, N + 1))  # (N+1, N+1), J[i,j] = x[i]
    # Actually let's do it properly:
    # We want D[i,j] = (c[i]/c[j]) / (x[i] - x[j]) for i != j
    # x_i - x_j: row i, col j
    x_col = x[:, None]  # (N+1, 1)
    x_row = x[None, :]  # (1, N+1)
    diff = x_col - x_row  # diff[i,j] = x[i] - x[j]
    c_col = c[:, None]  # (N+1, 1)
    c_row = c[None, :]  # (1, N+1)
    ratio = c_col / c_row  # ratio[i,j] = c[i]/c[j]

    # Avoid division by zero on diagonal
    D = np.zeros((N + 1, N + 1))
    mask = ~np.eye(N + 1, dtype=bool)
    D[mask] = ratio[mask] / diff[mask]
    # Diagonal: D[i,i] = -sum_{j != i} D[i,j]
    D[np.diag_indices(N + 1)] = -np.sum(D, axis=1) + D[np.diag_indices(N + 1)]
    # The diagonal is currently 0, so:
    D[np.diag_indices(N + 1)] = -np.sum(D, axis=1)
    return D, x


def integrate(f, t_span, y0):
    """
    Chebyshev spectral collocation for the Van der Pol oscillator.
    
    We solve for y at N+1 Chebyshev-Gauss-Lobatto nodes in [t0, t1].
    The ODE y' = f(t, y) is enforced at interior nodes (1..N-1).
    Boundary conditions: y(t0) = y0, and we use the ODE at t_end as well
    (or we can use N interior points + 1 boundary = N+1 equations).
    
    Actually for an IVP, we fix y[0] = y0 and enforce the ODE at nodes 1..N.
    That gives N+1 unknowns (y_1...y_N) with N equations from ODE + we need
    one more. Standard approach: enforce ODE at ALL nodes 0..N, but at node 0
    we use the initial condition instead. So we have N+1 equations:
    - y[0] = y0
    - D @ Y = f(t_i, y_i) for i = 1, ..., N
    where D is the derivative matrix scaled to [t0, t1].
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill

    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n_dim = len(y0)

    # Choose number of Chebyshev nodes
    # Van der Pol with mu=5 has somewhat stiff behavior but is smooth.
    # N=50 should be very accurate. Let's try a few sizes.
    for N in [40, 60, 80, 100]:
        if time.time() > deadline - 30.0:
            break

        try:
            # Get differentiation matrix on [-1, 1]
            D_x, x_nodes = _chebyshev_differential_matrix_fast(N)
            # Scale to [t0, t1]: dt/dx = (t1 - t0) / 2
            scale = (t1 - t0) / 2.0
            D_t = scale * D_x

            # Map nodes from [-1, 1] to [t0, t1]
            # x_nodes[0] = 1 -> t = t1, x_nodes[N] = -1 -> t = t0
            # t = (t0 + t1)/2 + (t1 - t0)/2 * x
            t_nodes = 0.5 * (t0 + t1) + 0.5 * (t1 - t0) * x_nodes
            # t_nodes[0] = t1, t_nodes[N] = t0
            # We want t_nodes[0] = t0, so reverse:
            t_nodes = t_nodes[::-1]
            D_t = D_t[::-1, :][::-1, :]  # Reverse rows and columns
            # Actually, if we reverse x, the derivative matrix transforms as:
            # If x' = -x, then d/dx' = -d/dx
            # Let me just rebuild properly.

            # Cleaner: define nodes from t0 to t1
            # x_i = cos(pi * i / N), i=0..N, gives x from 1 to -1
            # t_i = t0 + (t1 - t0) * (1 - x_i) / 2 = t0 + (t1-t0)/2 * (1 - x_i)
            # When x_i = 1 (i=0): t_i = t0
            # When x_i = -1 (i=N): t_i = t1
            # dt/dx_i = -(t1-t0)/2
            # So D_t = D_x * (-1) * (t1-t0)/2 ... no.
            # y'(t) = dy/dx * dx/dt
            # dx/dt = 1 / (dt/dx) = 1 / (-(t1-t0)/2) = -2/(t1-t0)
            # Wait: t = t0 + (t1-t0)/2 * (1 - x), so dt/dx = -(t1-t0)/2
            # dy/dt = dy/dx * dx/dt = dy/dx * (-2/(t1-t0))
            # So if D_x computes dy/dx, then dy/dt = -2/(t1-t0) * D_x @ y
            # Let me redefine: let s = (t - t0)/(t1-t0), s in [0,1]
            # t = t0 + (t1-t0)*s, dt/ds = t1-t0
            # dy/dt = dy/ds * ds/dt = dy/ds / (t1-t0)
            # s_i = (1 - x_i)/2 where x_i = cos(pi*i/N)
            # ds/dx = -1/2
            # dy/ds = dy/dx * dx/ds = dy/dx * (-2)
            # dy/dt = dy/dx * (-2) / (t1-t0)
            # So D_t = -2/(t1-t0) * D_x

            # Let me just use a direct approach with nodes sorted from t0 to t1.
            # Nodes: x_i = cos(pi*i/N) for i=0,...,N gives x from 1 to -1.
            # t_i = t0 + (t1-t0) * (1 - x_i)/2
            # t_0 = t0, t_N = t1. Good, sorted.
            x_nodes_orig = np.cos(np.pi * np.arange(N + 1) / N)  # 1 to -1
            t_nodes = t0 + (t1 - t0) * (1.0 - x_nodes_orig) / 2.0  # t0 to t1

            # Derivative matrix on [-1,1] for these nodes
            D_x, _ = _chebyshev_differential_matrix_fast(N)
            # dy/dt = -2/(t1-t0) * D_x @ y
            D_t = -2.0 / (t1 - t0) * D_x

            # Now set up the nonlinear system.
            # Unknowns: Y is (N+1) x n_dim matrix, Y[i] = y(t_i)
            # Constraints:
            #   Y[0] = y0  (initial condition)
            #   For i = 1, ..., N: (D_t @ Y)[:, i] = f(t_i, Y[i])  for each component
            # Total unknowns: (N+1)*n_dim, but Y[0] is fixed, so N*n_dim free unknowns.
            # Equations: N*n_dim (ODE at nodes 1..N)
            
            # Flatten: unknowns are Y[1], Y[2], ..., Y[N] (each n_dim)
            # Residual: R_i = (D_t @ Y)[:, i] - f(t_i, Y[i]) for i=1..N, for each component
            
            # Let's use Newton's method.
            # State: Y (N+1, n_dim), with Y[0] fixed.
            Y = np.zeros((N + 1, n_dim))
            Y[0] = y0
            # Initial guess: linear interpolation or constant
            for i in range(1, N + 1):
                frac = t_nodes[i] / t1 if t1 != 0 else 0
                Y[i] = y0  # Simple initial guess
            
            # Newton iteration
            max_newton = 30
            tol = 1e-13
            
            for newton_iter in range(max_newton):
                if time.time() > deadline - 10.0:
                    break
                
                # Compute residual
                # R = D_t @ Y - F, where F[i] = f(t_i, Y[i]) for i=1..N
                DY = D_t @ Y  # (N+1, n_dim)
                R = np.zeros((N, n_dim))
                for i in range(N):
                    R[i] = DY[i + 1] - f(t_nodes[i + 1], Y[i + 1])
                
                # Check convergence
                res_norm = np.linalg.norm(R, ord=np.inf)
                if res_norm < tol:
                    break
                
                # Compute Jacobian of the residual with respect to Y[1..N]
                # R_i = (D_t @ Y)[:, i+1] - f(t_{i+1}, Y[i+1])
                # dR_i/dY_j = D_t[i+1, j+1] * I - J_f(t_{i+1}, Y[i+1]) * delta_{i,j}
                # where J_f is the Jacobian of f.
                # For the Van der Pol system, we can compute J_f analytically.
                
                # Build the block Jacobian (N*n_dim) x (N*n_dim)
                # This is large for N=100, n_dim=2 -> 200x200. Manageable.
                # But for N=100, that's 100*2 = 200 unknowns, 200x200 Jacobian.
                
                # Actually, let's use a sparse-ish structure. The Jacobian is:
                # Block (i,j) = D_t[i+1, j+1] * I_n - J_f(t_{i+1}, Y[i+1]) * delta_{i,j}
                # So it's a dense matrix of blocks.
                
                # For efficiency, let's build it directly.
                J_full = np.zeros((N * n_dim, N * n_dim))
                for i in range(N):
                    for j in range(N):
                        block = D_t[i + 1, j + 1] * np.eye(n_dim)
                        if i == j:
                            # Subtract J_f
                            yi = Y[i + 1]
                            y0v, y1v = yi[0], yi[1]
                            Jf = np.array([[0.0, 1.0],
                                           [-10.0 * y0v * y1v - 1.0, 5.0 * (1.0 - y0v * y0v)]])
                            block -= Jf
                        J_full[i * n_dim:(i + 1) * n_dim, j * n_dim:(j + 1) * n_dim] = block
                
                # Solve J_full * delta = -R_flat
                R_flat = R.flatten()
                try:
                    delta = np.linalg.solve(J_full, -R_flat)
                except np.linalg.LinAlgError:
                    # Fallback: use least squares
                    delta, _, _, _ = np.linalg.lstsq(J_full, -R_flat, rcond=None)
                
                # Update Y[1..N]
                delta_Y = delta.reshape(N, n_dim)
                Y[1:] += delta_Y
                
                # Check for divergence
                if not np.all(np.isfinite(Y)):
                    Y[1:] -= delta_Y  # Revert
                    break

            # Check time
            if time.time() > deadline - 5.0:
                continue
            
            # Final residual check
            DY = D_t @ Y
            R = np.zeros((N, n_dim))
            for i in range(N):
                R[i] = DY[i + 1] - f(t_nodes[i + 1], Y[i + 1])
            res_norm = np.linalg.norm(R, ord=np.inf)
            
            if res_norm < 1e-10:
                # Return y at t_end = t_nodes[N]
                y_end = Y[N]
                if np.all(np.isfinite(y_end)):
                    return y_end
        
        except Exception:
            continue
    
    # Fallback: use solve_ivp if spectral method fails
    from scipy.integrate import solve_ivp
    
    def jac(t, y):
        y0v, y1v = y[0], y[1]
        return np.array([[0.0, 1.0],
                         [-10.0 * y0v * y1v - 1.0, 5.0 * (1.0 - y0v * y0v)]])
    
    sol = solve_ivp(f, t_span, y0, method='Radau', jac=jac,
                    rtol=1e-13, atol=1e-15, max_step=0.01)
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
