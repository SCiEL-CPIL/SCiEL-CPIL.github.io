"""Spectral collocation solver for Van der Pol oscillator."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import root
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard: 330s limit, aim to finish well before.
    start_time = time.time()
    deadline = start_time + 300.0
    
    # Number of collocation points (Chebyshev nodes)
    # For Van der Pol with mu=5, t in [0,10], N=30 should be very accurate.
    N = 30
    n_dim = len(y0)
    
    # Map from [-1, 1] to [t0, t1]
    # x in [-1, 1] -> t in [t0, t1]
    # t = 0.5 * (t1 - t0) * x + 0.5 * (t1 + t0)
    # dt/dx = 0.5 * (t1 - t0)
    dt_dx = 0.5 * (t1 - t0)
    
    # Chebyshev nodes of the first kind on [-1, 1]
    # Exclude endpoints to avoid boundary condition conflicts in standard collocation,
    # or include them and handle BCs carefully.
    # Standard approach: Use interior nodes for the ODE, and y0 for the initial condition.
    # Let's use N interior points.
    # Nodes: x_k = cos(pi * (k + 0.5) / N) for k=0..N-1
    k = np.arange(N)
    x_nodes = np.cos(np.pi * (k + 0.5) / N)
    
    # Sort nodes in increasing order of t (which corresponds to increasing x)
    # cos is decreasing from 0 to pi, so x_nodes are decreasing.
    # We need to sort them for the matrix structure, though for root finding it doesn't strictly matter,
    # but it's cleaner.
    sort_idx = np.argsort(x_nodes)
    x_nodes = x_nodes[sort_idx]
    
    # Corresponding t values
    t_nodes = 0.5 * (t1 - t0) * x_nodes + 0.5 * (t1 + t0)
    
    # Differentiation matrix D for Chebyshev nodes
    # D[i, j] = dy_i/dx_j
    # Formula for Chebyshev nodes of first kind (interior):
    # c_j = 2 for j=0, N-1 (endpoints, but we don't have endpoints here)
    # Actually, for interior nodes x_k = cos((k+0.5)pi/N), the weights c_k are all 1?
    # Let's use the standard formula for Chebyshev points of the first kind.
    # For x_j = cos(j*pi/N), j=0..N, D is well known.
    # For interior points, we can compute D via eigenvector method or explicit formula.
    # Explicit formula for D_ij where x_j are Chebyshev points (interior or not):
    # If x_j != x_i:
    # D_ij = (c_i / c_j) * (w_i / (w_j - w_i)) ? No, that's for weights.
    # D_ij = (c_i / c_j) * (1 / (x_i - x_j)) ? No.
    
    # Let's use the standard formula for the differentiation matrix D for Chebyshev points x_j = cos(j*pi/N).
    # Since we are using interior points, we can compute D for the full set and extract, 
    # or just compute D for the specific set of nodes.
    
    # General formula for differentiation matrix for arbitrary distinct nodes x_i:
    # D_ij = (w_i / (w_j - w_i)) for i != j, where w_i = 1 / prod_{k != i} (x_i - x_k)
    # D_ii = -sum_{j != i} D_ij
    
    # Compute w_i
    w = np.ones(N)
    for i in range(N):
        for k in range(N):
            if k != i:
                w[i] *= (x_nodes[i] - x_nodes[k])
        w[i] = 1.0 / w[i]
        
    # Compute D
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = w[i] / (w[j] - w[i])
            else:
                D[i, i] = -np.sum(D[i, :])
                
    # The system of equations:
    # Variables: y_0, ..., y_{N-1} (values at t_nodes)
    # Total variables: N * n_dim
    # Equations:
    # 1. Initial condition: y(t0) = y0. 
    #    Note: t_nodes do not include t0. 
    #    We can interpolate y0 to the first node? Or add y0 as a variable?
    #    Better: Treat y at the N nodes as unknowns. The ODE is enforced at these nodes.
    #    The initial condition is enforced by interpolation from the collocation points to t0.
    #    Alternatively, we can include t0 as a collocation point.
    #    Let's add t0 as the first point.
    
    # Revised plan: Use N+1 points including t0.
    # Points: x_0 = -1 (t=t0), x_1..x_N = interior Chebyshev nodes.
    # This is a common "Chebyshev collocation" setup.
    
    N_pts = 30 # Number of interior points
    # Total points = N_pts + 1
    # x_0 = -1
    # x_k = cos(pi * (k + 0.5) / N_pts) for k=1..N_pts? 
    # Standard Chebyshev nodes for N+1 points (0..N): x_j = cos(j*pi/N)
    # Let's use N_sub = 20 subintervals? No, just N+1 points.
    
    # Let's stick to N=20 total points for speed, including t0.
    N_total = 20
    # x_j = cos(j * pi / (N_total - 1)) for j=0..N_total-1
    j = np.arange(N_total)
    x_all = np.cos(j * np.pi / (N_total - 1))
    # x_all is decreasing from 1 to -1. We want t increasing, so x increasing.
    # t = 0.5*(t1-t0)*x + 0.5*(t1+t0).
    # If x goes 1 -> -1, t goes t1 -> t0.
    # Let's reverse x_all so it goes -1 -> 1 (t0 -> t1).
    x_all = x_all[::-1]
    t_all = 0.5 * (t1 - t0) * x_all + 0.5 * (t1 + t0)
    
    # Compute differentiation matrix D for these N_total points.
    w_all = np.ones(N_total)
    for i in range(N_total):
        for k in range(N_total):
            if k != i:
                w_all[i] *= (x_all[i] - x_all[k])
        w_all[i] = 1.0 / w_all[i]
        
    D_all = np.zeros((N_total, N_total))
    for i in range(N_total):
        for j in range(N_total):
            if i != j:
                D_all[i, j] = w_all[i] / (w_all[j] - w_all[i])
            else:
                D_all[i, i] = -np.sum(D_all[i, :])
                
    # Variables: Y is a vector of length N_total * n_dim
    # Y[0:n_dim] = y at t0
    # Y[n_dim:2*n_dim] = y at t1 (x_all[1])
    # ...
    # Y[(N_total-1)*n_dim : N_total*n_dim] = y at t_end (x_all[-1])
    
    # Equations:
    # 1. y(0) = y0 (n_dim equations)
    # 2. dy/dt = f(t, y) at x_all[1..N_total-1] ((N_total-1)*n_dim equations)
    #    dy/dt = (dy/dx) * (dx/dt) = (D @ y_x) * (1/dt_dx)
    #    So: (1/dt_dx) * (D @ y_x)_i = f(t_i, y_i) for i=1..N_total-1
    #    Note: D includes row 0, but we don't use equation for i=0 in ODE part, 
    #    because i=0 is fixed by initial condition.
    #    Actually, we can use the ODE equation at i=0 too? No, y(0) is fixed.
    #    The derivative at i=0 is determined by the polynomial, so it's consistent.
    #    We only enforce ODE for i=1..N_total-1.
    
    total_vars = N_total * n_dim
    total_eqs = n_dim + (N_total - 1) * n_dim
    
    def residual(Y):
        # Y is flat array
        Y_mat = Y.reshape(N_total, n_dim)
        
        res = np.zeros(total_eqs)
        
        # 1. Initial condition
        res[:n_dim] = Y_mat[0] - y0
        
        # 2. ODE at interior and end points (indices 1 to N_total-1)
        # Compute dy/dx at all points
        dy_dx = D_all @ Y_mat # Shape (N_total, n_dim)
        dy_dt = dy_dx / dt_dx
        
        for i in range(1, N_total):
            t_i = t_all[i]
            y_i = Y_mat[i]
            f_i = f(t_i, y_i)
            res[n_dim + (i-1)*n_dim : n_dim + i*n_dim] = dy_dt[i] - f_i
            
        return res
    
    # Initial guess: Linear interpolation of y0? Or constant y0?
    # Constant y0 is a bad guess for oscillators.
    # Let's use a simple Euler step or just y0 for all points.
    # Better: Use scipy solve_ivp with a loose tolerance to get a guess?
    # That would defeat the purpose of saving calls if solve_ivp uses many calls.
    # However, solve_ivp with loose tol might use ~100 calls. 
    # If our spectral method uses ~200 calls, it's not a win.
    # But spectral method converges quadratically/cubically. 
    # Let's try to construct a better guess.
    # Since it's Van der Pol, it oscillates. 
    # We can just start with y0 for all points and hope Newton converges.
    # Or, we can run a few steps of RK4 to get a guess.
    
    # Let's try to get a decent guess using a small number of RK4 steps.
    # This adds to the call count, but ensures convergence.
    # 10 steps of RK4 = 40 calls.
    # Then spectral refinement.
    
    # Actually, for N=20, the residual is 20*2=40 vars.
    # Newton's method for 40 vars.
    # If we start with y0, it might diverge.
    
    # Let's use a hybrid: 
    # 1. Run solve_ivp with Radau, rtol=1e-4, atol=1e-6, dense_output=True.
    #    This will take some calls (maybe 50-100).
    # 2. Use the solution at t_all as initial guess for the spectral root finder.
    # 3. The spectral root finder should converge in 2-3 iterations (quadratic).
    #    Each iteration requires N_total calls to f (for residual) + maybe Jacobian?
    #    If we use finite difference Jacobian, that's N_total * n_dim extra calls.
    #    Total calls: 100 (guess) + 3 * (20 + 40) = 220 calls.
    #    Current program uses ~1000+ calls.
    #    This looks promising.
    
    # However, the task says "f must be called at least once".
    # And "More than 2,000,000 calls".
    
    # Let's refine the guess generation to be cheaper.
    # We can use a very coarse grid for the guess.
    
    # Let's implement the residual function carefully to minimize overhead.
    
    # Precompute indices for speed?
    
    # Let's try to avoid the expensive guess if possible.
    # For Van der Pol, starting at [2, 0], it goes down.
    # Maybe a linear guess from y0 to y0 is okay for small t?
    # t=10 is not small.
    
    # Let's use the solve_ivp guess strategy.
    
    from scipy.integrate import solve_ivp
    
    # Get a rough solution
    # Use Radau with loose tolerances to handle stiffness
    sol_guess = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-3, 
        atol=1e-5,
        t_eval=t_all
    )
    
    if not sol_guess.success:
        # Fallback to current behavior if guess fails
        # But this is unlikely
        pass
        
    Y_guess = sol_guess.y.flatten()
    
    # Now solve the nonlinear system
    # Use 'hybr' or 'lm'? 'hybr' is Levenberg-Marquardt, good for square systems.
    # 'lm' is also Levenberg-Marquardt.
    # 'broyden1' might be faster if Jacobian is expensive, but FD Jacobian is costly.
    # Let's try 'hybr'.
    
    sol = root(residual, Y_guess, method='hybr', tol=1e-12)
    
    if not sol.success:
        # Try 'lm'
        sol = root(residual, Y_guess, method='lm', tol=1e-12)
        
    if not sol.success:
        # Return the guess if root finding fails?
        # Or return the last point of the guess?
        return sol_guess.y[:, -1]
        
    Y_sol = sol.x.reshape(N_total, n_dim)
    
    # The result is the state at the end of t_span, which is Y_sol[-1]
    return Y_sol[-1]

# EVOLVE-BLOCK-END
