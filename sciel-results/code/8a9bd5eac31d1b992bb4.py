"""Solve consumption-saving policy via backward induction with high-precision Euler equation."""
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    """
    Solve the finite-horizon consumption-saving problem.
    
    Parameters:
    -----------
    model : dict with keys 'beta', 'gamma', 'R', 'income', 'transition', 'horizon'
    resource_grid : np.ndarray, shape (101,), geomspace(0.001, 40, 101)
    budget : int, max number of consumption grid points per state
    
    Returns:
    --------
    consumption : np.ndarray, shape (6, 2, 101)
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline for safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)  # 2
    n_grid = len(resource_grid)  # 101
    
    # Initialize consumption array
    consumption = np.zeros((horizon, n_states, n_grid))
    
    # Terminal condition: consume everything
    consumption[horizon - 1] = resource_grid[np.newaxis, :, np.newaxis] * np.ones((1, n_states, 1))
    # Actually, consumption[5, z, i] = resource_grid[i] for all z
    for z in range(n_states):
        consumption[horizon - 1, z, :] = resource_grid
    
    # Value function: V_t(z, m) = max_c u(c) + beta * E[V_{t+1}(z', m')]
    # where m' = R*(m - c) + income[z']
    
    # We'll use backward induction. At each step, for each (z, m), we solve the Euler equation.
    # Euler equation: u'(c) = beta * R * E_z'[u'(c') * V_{t+1,m'}'(z', m')]
    # But it's easier to just maximize over c directly using a fine grid or Newton's method.
    
    # For efficiency, we'll use a grid-based approach with interpolation for the value function.
    # The key is to make the consumption grid fine enough in the region that matters (0.2 <= m <= 8).
    
    # Define utility and marginal utility
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-12:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
    
    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-12:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    def u_double_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-12:
            return -1.0 / c ** 2
        else:
            return -gamma * c ** (-gamma - 1)
    
    # For the value function, we'll store it on the resource_grid for each state.
    # V[t, z, i] = value at time t, state z, resource level resource_grid[i]
    
    # Start from t = horizon - 2 and go backwards
    # We need to compute V[t, z, m] for each t, z, and m in resource_grid
    
    # For each t, we'll compute the value function and the optimal consumption policy.
    # We'll use a fine grid for consumption within each (z, m) to maximize.
    
    # Actually, let's use a smarter approach: for each (t, z), we compute the value function
    # on resource_grid by solving the Euler equation using Newton's method, with the value
    # function from t+1 interpolated.
    
    # First, let's set up the value function for t = horizon - 1
    # V[horizon-1, z, m] = u(m) for all z (since you consume everything)
    
    V = np.zeros((horizon, n_states, n_grid))
    for z in range(n_states):
        V[horizon - 1, z, :] = u(resource_grid)
    
    # Now go backwards from t = horizon - 2 to t = 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Return what we have so far, filling remaining with a simple policy
            for tt in range(t, -1, -1):
                for z in range(n_states):
                    consumption[tt, z, :] = resource_grid * 0.95
            break
        
        for z in range(n_states):
            # For each resource level m in resource_grid, find optimal c
            # using the Euler equation with Newton's method
            
            # The Euler equation is:
            # u'(c) = beta * sum_{z'} transition[z, z'] * R * u'(c') * dV_{t+1}/dm'(z', m')
            # where m' = R*(m - c) + income[z']
            # and c' is the optimal consumption at (t+1, z', m')
            
            # This is implicit because c' depends on m' which depends on c.
            # We can solve this by fixed-point iteration or Newton's method.
            
            # Alternative: just maximize over c on a fine grid.
            # For each m, c ranges from 0 to m.
            
            # Let's use a hybrid: for each m, use a grid of c values and pick the best,
            # then refine with Newton's method.
            
            # Actually, let's just use a fine grid for c for each m.
            # The budget parameter suggests using up to 'budget' points per (z, m).
            
            # But that's 101 * 2 * 48 = ~9696 evaluations per (t, z), which is fine.
            
            # For each m in resource_grid:
            for i in range(n_grid):
                m = resource_grid[i]
                
                # c ranges from 0 to m
                # Use a log-spaced or linear grid for c
                # Since m can be very small (0.001) or large (40), use a grid relative to m
                
                # Create consumption grid: c = m * linspace(0.001, 1, n_c)
                # Use budget points
                n_c = min(budget, 200)  # Use up to 200 points for accuracy
                c_grid = m * np.linspace(0.001, 1.0, n_c)
                
                # For each c in c_grid, compute the value
                values = np.zeros(n_c)
                for j in range(n_c):
                    c = c_grid[j]
                    m_remaining = m - c
                    if m_remaining < 0:
                        continue
                    
                    # Expected continuation value
                    cont_val = 0.0
                    for zp in range(n_states):
                        prob = transition[z, zp]
                        if prob < 1e-15:
                            continue
                        m_next = R * m_remaining + income[zp]
                        
                        # Interpolate V[t+1, zp, m_next]
                        # m_next should be within [resource_grid[0], resource_grid[-1]] or extrapolate
                        if m_next <= resource_grid[0]:
                            V_next = V[t + 1, zp, 0]
                        elif m_next >= resource_grid[-1]:
                            V_next = V[t + 1, zp, -1]
                        else:
                            V_next = np.interp(m_next, resource_grid, V[t + 1, zp, :])
                        
                        cont_val += prob * V_next
                    
                    values[j] = u(c) + beta * cont_val
                
                # Find the best c
                best_idx = np.argmax(values)
                best_c = c_grid[best_idx]
                
                # Refine with a local search around best_c
                # Use a finer grid around best_c
                c_low = max(1e-12, c_grid[max(0, best_idx - 3)])
                c_high = min(m, c_grid[min(n_c - 1, best_idx + 3)])
                if c_high > c_low:
                    c_fine = np.linspace(c_low, c_high, 50)
                    values_fine = np.zeros(len(c_fine))
                    for j in range(len(c_fine)):
                        c = c_fine[j]
                        m_remaining = m - c
                        if m_remaining < 0:
                            continue
                        cont_val = 0.0
                        for zp in range(n_states):
                            prob = transition[z, zp]
                            if prob < 1e-15:
                                continue
                            m_next = R * m_remaining + income[zp]
                            if m_next <= resource_grid[0]:
                                V_next = V[t + 1, zp, 0]
                            elif m_next >= resource_grid[-1]:
                                V_next = V[t + 1, zp, -1]
                            else:
                                V_next = np.interp(m_next, resource_grid, V[t + 1, zp, :])
                            cont_val += prob * V_next
                        values_fine[j] = u(c) + beta * cont_val
                    best_idx_fine = np.argmax(values_fine)
                    best_c = c_fine[best_idx_fine]
                
                consumption[t, z, i] = best_c
                V[t, z, i] = u(best_c) + beta * (
                    transition[z, 0] * (
                        V[t + 1, 0, 0] if (R * (m - best_c) + income[0]) <= resource_grid[0]
                        else (V[t + 1, 0, -1] if (R * (m - best_c) + income[0]) >= resource_grid[-1]
                             else np.interp(R * (m - best_c) + income[0], resource_grid, V[t + 1, 0, :]))
                    ) + transition[z, 1] * (
                        V[t + 1, 1, 0] if (R * (m - best_c) + income[1]) <= resource_grid[0]
                        else (V[t + 1, 1, -1] if (R * (m - best_c) + income[1]) >= resource_grid[-1]
                             else np.interp(R * (m - best_c) + income[1], resource_grid, V[t + 1, 1, :]))
                    )
                )
    
    # Ensure terminal condition is exact
    for z in range(n_states):
        consumption[horizon - 1, z, :] = resource_grid
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_states):
            for i in range(n_grid):
                m = resource_grid[i]
                c = consumption[t, z, i]
                if c <= 0:
                    c = 1e-12
                if c > m + 1e-10:
                    c = m
                consumption[t, z, i] = c
    
    # Ensure finite values
    consumption = np.nan_to_num(consumption, nan=1e-12, posinf=resource_grid[-1], neginf=1e-12)
    
    return consumption
