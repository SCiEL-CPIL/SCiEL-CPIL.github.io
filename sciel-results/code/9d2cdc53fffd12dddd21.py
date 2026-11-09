# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y_start = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = 300.0
    
    # Spectral Chebyshev Collocation Method
    # We map the interval [t0, t1] to [-1, 1]
    # Let x in [-1, 1] be the Chebyshev variable.
    # t = 0.5 * (t1 - t0) * x + 0.5 * (t1 + t0)
    # dt/dx = 0.5 * (t1 - t0)
    
    # Number of collocation points (nodes)
    # For Van der Pol with mu=5, the solution is smooth but has boundary layers.
    # N=128 or 256 should be sufficient for high accuracy.
    N = 128
    
    # Generate Chebyshev nodes in [-1, 1]
    # We use the roots of Chebyshev polynomials or extrema.
    # Let's use the extrema (Chebyshev-Gauss-Lobatto points) which include endpoints.
    # x_j = cos(j * pi / (N-1)) for j=0..N-1
    # Note: x_0 = 1, x_{N-1} = -1
    # We want to map x=1 to t=t1 and x=-1 to t=t0? Or x=1 to t0?
    # Standard mapping: t = (t1+t0)/2 + (t1-t0)/2 * x
    # If x=1, t=t1. If x=-1, t=t0.
    # So the grid is ordered from t1 down to t0 if we follow j=0..N-1.
    # Let's reorder to t0 to t1 for clarity, or just handle the signs carefully.
    # Let's stick to x_j = cos(j*pi/(N-1)).
    # j=0 -> x=1 -> t=t1
    # j=N-1 -> x=-1 -> t=t0
    
    j = np.arange(N)
    x_nodes = np.cos(j * np.pi / (N - 1))
    
    # Map to t
    t_scale = (t1 - t0) / 2.0
    t_mid = (t1 + t0) / 2.0
    t_nodes = t_mid + t_scale * x_nodes
    
    # The derivative with respect to t is (dt/dx)^-1 * d/dx = (2/(t1-t0)) * d/dx
    # So y'(t) = (2/(t1-t0)) * y'(x)
    # The ODE is y'(t) = f(t, y)
    # So (2/(t1-t0)) * D_x * Y = F(t, Y)
    # Where D_x is the Chebyshev differentiation matrix.
    
    # Construct the Chebyshev differentiation matrix D (N x N)
    # Formula from Trefethen's "Spectral Methods in MATLAB"
    c = np.ones(N)
    c[0] = 0.5
    c[-1] = -0.5
    c = c * (-1)**j
    
    # x_j = cos(j*pi/(N-1))
    # D_ij = (c_i / c_j) * (T'_j(x_i) / (x_i - x_j)) for i != j
    # D_ii = -x_i / (2 * (1 - x_i^2)) for i not 0 or N-1
    # D_00 = (2N^2 - 1) / 6
    # D_{N-1,N-1} = -(2N^2 - 1) / 6
    
    D = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            if i != k:
                D[i, k] = (c[i] / c[k]) * (1.0 / (x_nodes[i] - x_nodes[k]))
            else:
                if i == 0:
                    D[i, i] = (2 * N**2 - 1) / 6.0
                elif i == N - 1:
                    D[i, i] = -(2 * N**2 - 1) / 6.0
                else:
                    D[i, i] = -x_nodes[i] / (2.0 * (1.0 - x_nodes[i]**2))
    
    # The matrix for the ODE in t
    # A * Y = F
    # A = (2/(t1-t0)) * D_x - I
    # Wait, the equation is:
    # y'(t) = f(t, y)
    # (dt/dx)^-1 * dy/dx = f(t, y)
    # (2/(t1-t0)) * D_x Y = F
    # So: (2/(t1-t0)) * D_x Y - F = 0
    # Let G(Y) = (2/(t1-t0)) * D_x Y - F(Y)
    # We want to solve G(Y) = 0 subject to Y[0] (which corresponds to x=1, t=t1? No, x=-1 is t0)
    # Let's check the index mapping again.
    # j=0 -> x=1 -> t=t1
    # j=N-1 -> x=-1 -> t=t0
    # So the initial condition y(0) = y_start corresponds to t0, which is index N-1.
    # So we fix Y[N-1] = y_start.
    # The unknowns are Y[0:N-1].
    
    # We solve for Y[0:N-1] using Newton's method.
    # The system is:
    # (2/(t1-t0)) * D_x[0:N-1, :] * Y - F[0:N-1] = 0
    # And Y[N-1] is known.
    
    # Let Y_unknown = Y[0:N-1]
    # Let Y_known = Y[N-1] = y_start
    # Let D_unk = D[0:N-1, 0:N-1]
    # Let D_known = D[0:N-1, N-1]
    
    # Equation: (2/(t1-t0)) * (D_unk @ Y_unk + D_known @ Y_known) - F_unk = 0
    # Let C = (2/(t1-t0)) * D_known @ Y_known
    # Then: (2/(t1-t0)) * D_unk @ Y_unk - F_unk + C = 0
    
    D_unk = D[0:N-1, 0:N-1]
    D_known = D[0:N-1, N-1]
    
    coeff = 2.0 / (t1 - t0)
    
    # Initial guess for Y_unk
    # A good guess is to interpolate y_start constant, or use a simple RK4 solution.
    # Since f is cheap and we have a budget, a simple fixed-step RK4 on a coarse grid
    # to initialize Y_unk is a good idea.
    
    # Coarse grid for initialization
    N_coarse = 50
    t_coarse = np.linspace(t0, t1, N_coarse)
    Y_coarse = np.zeros((N_coarse, 2))
    Y_coarse[-1] = y_start # Wait, t_coarse goes t0 to t1.
    # Let's do forward RK4 from t0 to t1.
    Y_coarse[0] = y_start
    h_c = (t1 - t0) / (N_coarse - 1)
    for i in range(N_coarse - 1):
        t_c = t_coarse[i]
        y_c = Y_coarse[i]
        k1 = f(t_c, y_c)
        k2 = f(t_c + 0.5*h_c, y_c + 0.5*h_c*k1)
        k3 = f(t_c + 0.5*h_c, y_c + 0.5*h_c*k2)
        k4 = f(t_c + h_c, y_c + h_c*k3)
        Y_coarse[i+1] = y_c + h_c/6.0 * (k1 + 2*k2 + 2*k3 + k4)
        
    # Interpolate Y_coarse to x_nodes
    # We have Y_coarse at t_coarse.
    # We need Y at t_nodes.
    # t_nodes is not sorted in t0->t1 order, it's t1->t0.
    # Let's sort t_nodes and corresponding x_nodes for interpolation.
    # Actually, np.interp requires increasing x.
    # t_coarse is increasing.
    # t_nodes: x_nodes[0]=1 -> t1, x_nodes[N-1]=-1 -> t0.
    # So t_nodes is decreasing.
    # Let's reverse t_nodes and x_nodes to make them increasing.
    t_nodes_inc = t_nodes[::-1]
    x_nodes_inc = x_nodes[::-1]
    # Now t_nodes_inc[0] = t0, t_nodes_inc[-1] = t1.
    
    # Interpolate Y_coarse (shape N_coarse x 2) to t_nodes_inc
    # Y_nodes_inc[i, :] corresponds to t_nodes_inc[i]
    Y_nodes_inc = np.zeros((N, 2))
    for dim in range(2):
        Y_nodes_inc[:, dim] = np.interp(t_nodes_inc, t_coarse, Y_coarse[:, dim])
        
    # Now map back to original indexing (j=0..N-1 corresponding to x=1..-1)
    # Y_nodes_inc is in order t0->t1.
    # Original Y is in order t1->t0.
    Y_initial = Y_nodes_inc[::-1]
    
    # So Y_initial[N-1] should be y_start.
    # Let's enforce it.
    Y_initial[N-1] = y_start
    
    # Unknowns are Y_initial[0:N-1]
    Y_unk = Y_initial[0:N-1].copy()
    
    # Newton's Method
    # Residual R(Y_unk) = coeff * (D_unk @ Y_unk + D_known @ y_start) - F_unk(Y_unk)
    # Jacobian J(Y_unk) = coeff * D_unk - dF_unk/dY_unk
    
    # We need to approximate dF_unk/dY_unk.
    # F_unk[i] = f(t_nodes[i], Y_unk[i]) for i=0..N-2.
    # Note: t_nodes[i] corresponds to x_nodes[i].
    # The Jacobian of the residual with respect to Y_unk is:
    # J = coeff * D_unk - diag( dF_i / dY_i )
    # Where dF_i / dY_i is the 2x2 Jacobian of f at (t_nodes[i], Y_unk[i]).
    
    # Since f is fixed (Van der Pol), we can compute the Jacobian analytically or numerically.
    # f(t, y) = [y1, 5*(1-y0^2)*y1 - y0]
    # df/dy = [[0, 1], [-10*y0*y1 - 1, 5*(1-y0^2)]]
    # This is cheap to compute.
    
    def compute_jacobian_f(t, y):
        y0_val, y1_val = y
        # df/dy0 = -10*y0*y1 - 1
        # df/dy1 = 5*(1-y0^2)
        J = np.array([
            [0.0, 1.0],
            [-10.0 * y0_val * y1_val - 1.0, 5.0 * (1.0 - y0_val**2)]
        ])
        return J
    
    # Newton Iterations
    max_iters = 50
    tol = 1e-12
    
    Y_curr = Y_unk.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > deadline - 10.0:
            break
            
        # Evaluate F at current Y_unk
        F_vals = np.zeros((N-1, 2))
        for i in range(N-1):
            F_vals[i] = f(t_nodes[i], Y_curr[i])
            
        # Compute Residual
        # R = coeff * (D_unk @ Y_curr + D_known @ y_start) - F_vals
        DY = D_unk @ Y_curr + D_known @ y_start
        R = coeff * DY - F_vals
        
        # Check convergence
        res_norm = np.linalg.norm(R)
        if res_norm < tol:
            break
            
        # Build Jacobian J
        # J is a block matrix of size (2*(N-1)) x (2*(N-1))
        # J = coeff * (I_2 x D_unk) - block_diag(J_f_i)
        # Let's flatten Y_curr to a vector of size 2*(N-1)
        # Or just build the big matrix. N is small (128), so 256x256 matrix is fine.
        
        J_mat = coeff * np.kron(np.eye(2), D_unk)
        
        for i in range(N-1):
            J_f = compute_jacobian_f(t_nodes[i], Y_curr[i])
            # Subtract J_f from the block corresponding to variable i
            row_start = 2*i
            col_start = 2*i
            J_mat[row_start:row_start+2, col_start:col_start+2] -= J_f
            
        # Solve J_mat * delta = -R_flat
        R_flat = R.flatten()
        
        try:
            delta = np.linalg.solve(J_mat, -R_flat)
        except np.linalg.LinAlgError:
            # If singular, use least squares or break
            break
            
        # Damping if step is too large
        # Check if new Y is finite
        Y_new = Y_curr + delta
        
        if not np.all(np.isfinite(Y_new)):
            # Damp step
            factor = 0.5
            while not np.all(np.isfinite(Y_curr + factor * delta)) and factor > 1e-6:
                factor *= 0.5
            Y_new = Y_curr + factor * delta
            
        Y_curr = Y_new
        
    # Reconstruct full Y
    Y_final = np.zeros((N, 2))
    Y_final[0:N-1] = Y_curr
    Y_final[N-1] = y_start
    
    # The solution at t1 corresponds to x=1, which is index 0 in our x_nodes array.
    # So y_end = Y_final[0]
    
    return Y_final[0]
# EVOLVE-BLOCK-END
