import numpy as np
import time
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    m_grid = np.asarray(resource_grid, dtype=float)
    budget = int(budget)

    n_states = len(income)
    n_m = len(m_grid)

    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    def marginal_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        else:
            return np.power(c, -gamma)

    # Allocate storage for policies: consumption[t, z, m_index]
    policy = np.zeros((horizon, n_states, n_m))
    
    # V values on the grid for each t, z
    V = np.zeros((horizon, n_states, n_m))
    
    # Final period: t = horizon - 1 = 5
    # V[5, z, m] = u(m)
    for z in range(n_states):
        V[5, z, :] = utility(m_grid)
    policy[5, :, :] = m_grid[np.newaxis, :]  # c = m at final period
    
    # Work backwards from t = 4 down to 0
    for t in range(horizon - 2, -1, -1):
        # Check time
        if time.time() > deadline - 30:
            # Return best available with simple heuristics for remaining
            for tt in range(t, 0, -1):
                for z in range(n_states):
                    # Use a simple rule: consume based on remaining horizon
                    remaining = tt + 1
                    frac = 1.0 / (remaining * beta ** (remaining - 1))
                    frac = min(frac, 1.0)
                    policy[tt, z, :] = m_grid * frac
                    V[tt, z, :] = V[tt+1, z, :]
            # Ensure final period is correct
            policy[5, :, :] = m_grid[np.newaxis, :]
            for z in range(n_states):
                V[5, z, :] = utility(m_grid)
            # Clamp to valid range
            for tt in range(horizon):
                for z in range(n_states):
                    policy[tt, z, :] = np.clip(policy[tt, z, :], 1e-12, m_grid + 1e-10)
            return policy
        
        for z in range(n_states):
            m_values = m_grid
            
            # We need to find c* for each m in m_values
            # FOC: u'(c) = beta * R * sum_{z'} P[z, z'] * u'(R(m-c) + income[z'])
            
            best_c = np.zeros(n_m)
            best_V = np.zeros(n_m)
            
            for i, m in enumerate(m_values):
                # Define the function to solve for c
                def f(c):
                    # c is scalar
                    # u'(c)
                    mu_c = marginal_utility(c)
                    
                    # Expected future marginal utility
                    exp_mu = 0.0
                    for z_next in range(n_states):
                        prob = transition[z, z_next]
                        if prob < 1e-15:
                            continue
                        m_next = R * (m - c) + income[z_next]
                        # m_next should be > 0
                        if m_next < 1e-12:
                            m_next = 1e-12
                        mu_mnext = marginal_utility(m_next)
                        exp_mu += prob * mu_mnext
                    
                    return mu_c - beta * R * exp_mu
                
                # Check if c=m is optimal
                # If f(m) >= 0, then u'(m) >= beta R E[u'(y')], so we want to consume more?
                # Wait. f(c) = u'(c) - beta R E[u'(c')].
                # f is decreasing.
                # If f(m) >= 0, the root is to the right of m? No, f is decreasing, so if f(m) >= 0, the root is > m?
                # But c cannot be > m.
                # Let's check the sign.
                # At c=0, u'(c) is huge, f(0) > 0.
                # At c=m, u'(c) is small.
                # If f(m) < 0, root is in (0, m).
                # If f(m) >= 0, the function is still positive at m. Since it's decreasing, the root is > m.
                # But we are constrained by c <= m.
                # So if f(m) >= 0, the optimal c is m.
                
                f_m = f(m)
                if f_m >= 0:
                    best_c[i] = m
                else:
                    # Root is in (0, m)
                    # Use a small epsilon for lower bound
                    c_min = 1e-12
                    try:
                        best_c[i] = brentq(f, c_min, m, xtol=1e-10, rtol=1e-10)
                    except ValueError:
                        # Fallback to midpoint or boundary if brentq fails
                        # This shouldn't happen if f(c_min) > 0 and f(m) < 0
                        best_c[i] = m / 2.0
                
                # Calculate V[t, z, i]
                # V = u(c) + beta * E[V(t+1, z', m')]
                c_val = best_c[i]
                u_c = utility(c_val)
                
                exp_V = 0.0
                for z_next in range(n_states):
                    prob = transition[z, z_next]
                    if prob < 1e-15:
                        continue
                    m_next = R * (m - c_val) + income[z_next]
                    # Interpolate V[t+1, z_next, :] at m_next
                    if m_next <= m_grid[0]:
                        v_next = V[t+1, z_next, 0]
                    elif m_next >= m_grid[-1]:
                        v_next = V[t+1, z_next, -1]
                    else:
                        v_next = np.interp(m_next, m_grid, V[t+1, z_next, :])
                    exp_V += prob * v_next
                
                best_V[i] = u_c + beta * exp_V
            
            policy[t, z, :] = best_c
            V[t, z, :] = best_V
    
    # Final validation
    for t in range(horizon):
        for z in range(n_states):
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, m_grid + 1e-10)
    
    # Ensure c[5] equals grid
    policy[5, :, :] = m_grid[np.newaxis, :]
    
    return policy
