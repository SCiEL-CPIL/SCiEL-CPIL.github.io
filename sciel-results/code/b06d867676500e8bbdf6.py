import numpy as np

def solve_consumption(model, resource_grid, budget):
    """
    Solve the finite-horizon consumption-saving problem using Chebyshev
    spectral collocation for the value function approximation.
    
    Parameters:
    -----------
    model : dict with keys 'beta', 'gamma', 'R', 'income', 'transition', 'horizon'
    resource_grid : np.array of shape (101,) - the grid of resource levels
    budget : int - number of points in the seed's backward induction (unused here)
    
    Returns:
    --------
    consumption : np.array of shape (6, 2, 101)
    """
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = 2  # number of income states
    n_grid = len(resource_grid)
    
    # Define the utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    # Chebyshev nodes and weights for spectral approximation
    # We'll use Chebyshev nodes on [-1, 1] mapped to the resource grid
    # But actually, for better accuracy, we can use the given grid directly
    # with Chebyshev interpolation on a transformed scale
    
    # For CRRA with gamma != 1, the value function is smooth but may have
    # different behavior at different scales. Let's use a log transformation
    # of the resource variable to handle the wide range better.
    
    # Actually, let's use Chebyshev interpolation on the original grid.
    # The grid is geomspace, so it's already spread out nicely.
    
    # Number of Chebyshev nodes - use all 101 points for high-order approximation
    # Chebyshev nodes of first kind: x_k = cos(pi * k / (N-1)) for k=0,...,N-1
    # But we want to use the given grid, so let's map the grid to [-1, 1]
    
    # Map resource_grid to [-1, 1]
    m_min = resource_grid[0]
    m_max = resource_grid[-1]
    # Use log mapping for better resolution at small m
    log_m_min = np.log(m_min)
    log_m_max = np.log(m_max)
    log_grid = np.log(resource_grid)
    # Map to [-1, 1]
    x_map = (log_grid - log_m_min) / (log_m_max - log_m_min) * 2 - 1
    
    # Chebyshev interpolation: given values f(x_i) at Chebyshev nodes,
    # compute coefficients using DCT
    # But our x_map are not exactly Chebyshev nodes. Let's use the standard
    # approach: interpolate on Chebyshev nodes, then evaluate.
    
    # Better approach: use the 101 grid points as collocation points and
    # fit a Chebyshev polynomial of degree up to 100.
    
    # For each time step and state, we have V_t(m, z) as a function of m.
    # We'll represent V_t(m, z) as a Chebyshev polynomial in the mapped variable.
    
    # Build the Chebyshev basis matrix
    # T_k(x) = cos(k * arccos(x))
    N = n_grid  # 101
    max_degree = N - 1  # degree 100
    
    # Compute Chebyshev values at the grid points
    theta = np.arccos(x_map)  # shape (N,)
    # T_k(x_i) = cos(k * theta_i)
    k_vals = np.arange(max_degree + 1)  # 0 to 100
    # Cheb_vals[i, k] = cos(k * theta_i)
    Cheb_vals = np.cos(np.outer(k_vals, theta))  # shape (N, N)
    
    # Inverse for interpolation: given function values f at grid points,
    # find Chebyshev coefficients c such that f_i ≈ sum_k c_k T_k(x_i)
    # This is a linear system: Cheb_vals.T @ c = f
    # Cheb_vals is N x N, so we can invert it.
    
    # Precompute the inverse
    try:
        Cheb_inv = np.linalg.inv(Cheb_vals)
    except:
        Cheb_inv = np.linalg.pinv(Cheb_vals)
    
    # Now solve by backward induction
    # V_t(m, z) = max_c [ u(c) + beta * E_z'[V_{t+1}(m', z')] ]
    # where m' = R * (m - c) + income[z']
    
    # Terminal condition: V_H(m, z) = u(m) (consume everything at the end)
    # Actually, at t = horizon-1 (the last period), you consume all resources.
    # So V_{horizon}(m, z) = 0, and at t = horizon-1:
    # V_{horizon-1}(m, z) = u(m) + beta * E[0] = u(m)
    
    # Let's define:
    # V[t][z] is an array of shape (N,) giving V_t(m_i, z) for i=0,...,N-1
    # c[t][z] is the optimal consumption at each grid point
    
    V = np.zeros((horizon + 1, n_states, N))
    c_opt = np.zeros((horizon, n_states, N))
    
    # Terminal: V[horizon][z][i] = 0
    # For t = horizon-1 down to 0:
    for t in range(horizon - 1, -1, -1):
        for z in range(n_states):
            # For each grid point m_i, find optimal c
            m_i = resource_grid
            
            # We need to maximize: u(c) + beta * sum_{z'} P(z'|z) * V[t+1, z'](m')
            # where m' = R * (m_i - c) + income[z']
            
            # Since we're using spectral approximation, we can evaluate V[t+1, z']
            # at any m' by interpolating the Chebyshev polynomial.
            
            # First, get the Chebyshev coefficients for V[t+1, z']
            # V[t+1, z'] is stored as values at the grid points.
            # Coefficients c_k such that V[m] ≈ sum_k c_k T_k(x(m))
            
            # For each z', compute the continuation value as a function of m'
            # Then for each m_i, optimize over c.
            
            # Let's use a fine grid for c to find the optimum
            # c in [0, m_i], but we need c > 0 for log utility
            # Use a logarithmic grid for c to handle small values well
            
            # For each m_i, the optimal c satisfies the Euler equation:
            # u'(c) = beta * R * E_z'[V_{t+1, z}'(m')]
            # where m' = R*(m_i - c) + income[z']
            
            # This is a 1D root-finding problem. Let's use a grid search
            # with fine resolution, then refine.
            
            # Actually, for CRRA utility, the Euler equation is:
            # c^(-gamma) = beta * R * E_z'[V_{t+1, z}'(m')]
            # (for gamma != 1; for gamma=1, 1/c = beta * R * E[...])
            
            # Let's use a numerical approach: for each m_i, maximize the objective
            # over c using a fine grid.
            
            # To speed up, we can use the fact that the optimal c is monotone in m.
            # But let's just do a fine grid search for now.
            
            # Number of c points for optimization
            n_c = 200  # fine grid for c
            
            # For each m_i, c ranges from small to m_i
            # Let's use a log-spaced grid for c relative to m_i
            
            # Precompute the Chebyshev coefficients for V[t+1, z']
            V_next_coeffs = np.zeros((n_states, N))
            for z_prime in range(n_states):
                # V[t+1, z_prime] is values at grid points
                V_next = V[t + 1, z_prime]
                # Find Chebyshev coefficients
                V_next_coeffs[z_prime] = Cheb_inv @ V_next
            
            # Function to evaluate V[t+1, z'](m) for any m
            def eval_V_next(m_eval, z_prime):
                """Evaluate V[t+1, z_prime] at resource level m_eval."""
                # Map m_eval to x in [-1, 1]
                log_m_eval = np.log(np.maximum(m_eval, 1e-15))
                x_eval = (log_m_eval - log_m_min) / (log_m_max - log_m_min) * 2 - 1
                x_eval = np.clip(x_eval, -1.0, 1.0)
                
                # Evaluate Chebyshev polynomial
                theta_eval = np.arccos(x_eval)
                # T_k(x_eval) = cos(k * theta_eval)
                k_vals_eval = np.arange(N)
                T_vals = np.cos(np.outer(k_vals_eval, theta_eval))  # shape (N, len(m_eval))
                # V = sum_k c_k T_k
                V_eval = T_vals.T @ V_next_coeffs[z_prime]
                return V_eval
            
            # For each m_i, find optimal c
            for i in range(N):
                m_i_val = resource_grid[i]
                
                # Grid for c: from 1e-12 to m_i_val
                # Use a combination of log and linear spacing
                c_grid = np.linspace(1e-12, m_i_val, n_c)
                
                # Compute objective for each c
                obj = np.zeros(n_c)
                for j in range(n_c):
                    c_j = c_grid[j]
                    u_c = utility(c_j)
                    
                    # Compute continuation value
                    cont_val = 0.0
                    for z_prime in range(n_states):
                        prob = transition[z, z_prime]
                        m_prime = R * (m_i_val - c_j) + income[z_prime]
                        m_prime = max(m_prime, 1e-12)
                        V_next_val = eval_V_next(m_prime, z_prime)
                        cont_val += prob * V_next_val
                    
                    obj[j] = u_c + beta * cont_val
                
                # Find the maximum
                best_idx = np.argmax(obj)
                c_opt[t, z, i] = c_grid[best_idx]
                V[t, z, i] = obj[best_idx]
    
    # Terminal period: consume everything
    c_opt[horizon - 1, :, :] = resource_grid[None, :]
    # Actually, at t = horizon-1, V = u(m), and c = m
    # Let's recompute the last period properly
    for z in range(n_states):
        c_opt[horizon - 1, z, :] = resource_grid
        V[horizon - 1, z, :] = utility(resource_grid)
    
    # Ensure constraints: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_states):
            c_opt[t, z] = np.maximum(c_opt[t, z], 1e-12)
            c_opt[t, z] = np.minimum(c_opt[t, z], resource_grid + 1e-10)
    
    # Ensure c[5] (last period) equals the grid within 1e-9
    c_opt[horizon - 1, :, :] = resource_grid[None, :]
    
    return c_opt
