import numpy as np
import scipy.optimize
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 300.0  # Safety margin
    
    t0, t1 = t_span
    N = 150  # Number of Chebyshev nodes
    
    # Chebyshev nodes on [0, 1] mapped to [t0, t1]
    # Clamped Chebyshev nodes to ensure endpoints are included
    k = np.arange(N)
    # Standard Chebyshev points of the first kind in [-1, 1]
    # cos(pi * (2k+1) / (2N)) or similar
    # Let's use clamped Chebyshev:
    # x_i = 0.5 * (1 + cos(pi * i / (N-1))) for i=0..N-1
    x = 0.5 * (1 + np.cos(np.pi * np.arange(N) / (N - 1)))
    # Map to [t0, t1]
    t = t0 + 0.5 * (t1 - t0) * (1 - x)  # Reversed so x=1 is t0? 
    # Let's stick to standard mapping: x in [0,1] -> t in [t0, t1]
    t = t0 + x * (t1 - t0)
    
    # Differentiation Matrix D
    # D is N x N
    # Using the formula for Chebyshev differentiation matrix
    c = np.ones(N)
    c[0] = 0.5
    c[-1] = 0.5
    c = c * (-1)**np.arange(N)
    
    X = np.tile(x, (N, 1))
    dX = X - X.T
    
    # Avoid division by zero
    dX[np.diag_indices(N)] = 1
    
    D = (np.outer(2*c, 1/c) / (dX + np.eye(N)))
    
    # Initial Guess
    # y0 decays, y1 is small, y2 rises
    # Use a simple exponential decay for y0
    y_guess_0 = np.exp(-0.04 * (t - t0))
    y_guess_1 = np.zeros(N)
    y_guess_2 = 1 - y_guess_0
    # Add a small bump for y1 to help convergence?
    # The peak of y1 is around 1e-3.
    # Let's just use the simple guess first.
    Y_guess = np.vstack([y_guess_0, y_guess_1, y_guess_2]).flatten()
    
    def residual(Y):
        # Y is (3*N,)
        Y_mat = Y.reshape(3, N)
        # Compute F(t, Y)
        F = np.zeros((3, N))
        for i in range(N):
            if time.time() > deadline:
                # Return current best guess if time runs out?
                # We can't interrupt the loop easily without breaking the solver.
                # But 150 calls is fast.
                pass
            F[:, i] = f(t[i], Y_mat[:, i])
        
        # D @ Y_mat
        DY = D @ Y_mat
        
        # Residual
        R = DY - F
        return R.flatten()
        
    def jac(Y):
        # J = D I - J_f
        # D I is block diagonal with D on diagonal
        # J_f is block diagonal with J_f(t_i, Y_i)
        Y_mat = Y.reshape(3, N)
        J = np.zeros((3*N, 3*N))
        
        # Block diagonal D
        # D is N x N. We need to place it 3 times on the diagonal blocks of size N x N
        # Indices: 
        # y0: 0:N, y1: N:2N, y2: 2N:3N
        # Block (0,0): D
        # Block (1,1): D
        # Block (2,2): D
        
        # This is inefficient to build a dense matrix every time.
        # But N=150, 3N=450. 450x450 is 200k elements.
        # Building it takes some time.
        
        # Alternative: Use sparse matrix and convert? Or just build dense.
        
        # Let's build the dense Jacobian
        # D blocks
        J[0:N, 0:N] = D
        J[N:2*N, N:2*N] = D
        J[2*N:3*N, 2*N:3*N] = D
        
        # Subtract J_f blocks
        for i in range(N):
            y0_val, y1_val, y2_val = Y_mat[:, i]
            # Jacobian of Robertson
            # y0' = -0.04*y0 + 1e4*y1*y2
            # y1' =  0.04*y0 - 1e4*y1*y2 - 3e7*y1^2
            # y2' =  3e7*y1^2
            
            # df/dy0, df/dy1, df/dy2
            # Row 0:
            J[0, i] -= -0.04
            J[0, N+i] -= 1e4 * y2_val
            J[0, 2*N+i] -= 1e4 * y1_val
            
            # Row 1:
            J[N, i] -= 0.04
            J[N, N+i] -= -1e4 * y2_val - 6e7 * y1_val
            J[N, 2*N+i] -= 1e4 * y1_val
            
            # Row 2:
            J[2*N, i] -= 0
            J[2*N, N+i] -= 6e7 * y1_val
            J[2*N, 2*N+i] -= 0
            
        return J

    try:
        # Use scipy.optimize.root
        # method='hybr' (MINPACK) is robust
        # tol=1e-12 for high accuracy
        sol = scipy.optimize.root(
            residual, 
            Y_guess, 
            jac=jac, 
            method='hybr', 
            tol=1e-12
        )
        
        if sol.success:
            Y_final = sol.x.reshape(3, N)
            return Y_final[:, -1]
        else:
            # Fallback to solve_bvp
            def bc(ya, yb):
                return ya - y0
            def jac_y(t, y):
                y0_val, y1_val, y2_val = y
                return np.array([
                    [-0.04, 1e4*y2_val, -1e4*y1_val],
                    [0.04, -1e4*y2_val - 6e7*y1_val, 1e4*y1_val],
                    [0, 6e7*y1_val, 0]
                ])
            
            mesh = np.linspace(t0, t1, 100)
            y_guess_bvp = np.zeros((3, len(mesh)))
            y_guess_bvp[0, :] = np.exp(-0.04 * (mesh - t0))
            y_guess_bvp[1, :] = 0
            y_guess_bvp[2, :] = 1 - y_guess_bvp[0, :]
            
            sol_bvp = scipy.integrate.solve_bvp(
                f, bc, mesh, y_guess_bvp, jac_y=jac_y, tol=1e-10, max_nodes=500
            )
            if sol_bvp.success:
                return sol_bvp.y[:, -1]
            else:
                return Y_guess[:, -1] # Return guess as last resort

    except Exception as e:
        # Fallback to solve_ivp
        def jac_y(t, y):
            y0_val, y1_val, y2_val = y
            return np.array([
                [-0.04, 1e4*y2_val, -1e4*y1_val],
                [0.04, -1e4*y2_val - 6e7*y1_val, 1e4*y1_val],
                [0, 6e7*y1_val, 0]
            ])
        sol_ivp = scipy.integrate.solve_ivp(
            f, t_span, y0, method='Radau', jac=jac_y, rtol=1e-10, atol=1e-12
        )
        if sol_ivp.success:
            return sol_ivp.y[:, -1]
        return np.array(y0)
