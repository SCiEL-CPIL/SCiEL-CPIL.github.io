import numpy as np
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    T = model['horizon']
    
    n_grid = len(resource_grid)
    n_states = len(income)
    
    c_policy = np.zeros((T, n_states, n_grid))
    V = np.zeros((T, n_states, n_grid))
    
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
            
    def marginal_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    # Terminal period
    c_policy[T-1, :, :] = resource_grid[None, None, :]
    V[T-1, :, :] = utility(resource_grid)[None, :]
    
    # Backward induction
    for t in range(T - 2, -1, -1):
        for z in range(n_states):
            for i, m in enumerate(resource_grid):
                # Define the Euler equation residual
                def euler_residual(c, m_val=m, t_val=t, z_val=z):
                    # c is the current consumption
                    # m_val is the current wealth
                    # Calculate next wealths
                    next_wealths = R * (m_val - c) + income
                    # Next consumptions via interpolation
                    # c_policy[t+1, zp, :] is the policy for next period
                    next_cs = np.zeros(n_states)
                    for zp in range(n_states):
                        # Clip next_wealth to grid range for interpolation? 
                        # np.interp handles extrapolation by returning edge values if not specified, 
                        # but default is to return first/last value. 
                        # However, wealth can be negative? 
                        # m >= 0, c <= m => m-c >= 0. R=1.02. Income > 0. 
                        # So next_wealth >= min(income) > 0.
                        # Grid starts at 0.001.
                        # If next_wealth < grid[0], interp returns grid[0] value? 
                        # Actually, np.interp(x, xp, fp) returns fp[0] if x < xp[0].
                        # This is a reasonable approximation.
                        next_cs[zp] = np.interp(next_wealths[zp], resource_grid, c_policy[t+1, zp, :])
                    
                    # Expected marginal utility of next consumption
                    exp_mu_next = 0.0
                    for zp in range(n_states):
                        exp_mu_next += transition[z_val, zp] * marginal_utility(next_cs[zp])
                    
                    return marginal_utility(c) - beta * R * exp_mu_next

                # Solve for c
                # Check if consuming all is optimal
                # If euler_residual(m) > 0, it means u'(m) > beta R E[u'(c_next)], 
                # so we want to consume more, but we can't. So c = m.
                res_m = euler_residual(m)
                if res_m > 0:
                    c_opt = m
                else:
                    # Find root in [1e-12, m]
                    # We know res(1e-12) > 0 because u'(1e-12) is huge.
                    # We know res(m) <= 0.
                    try:
                        c_opt = brentq(euler_residual, 1e-12, m, xtol=1e-12, rtol=1e-12)
                    except ValueError:
                        # Fallback if no sign change (shouldn't happen if res_m <= 0)
                        c_opt = m 
                        
                # Ensure constraints
                if c_opt < 1e-12:
                    c_opt = 1e-12
                if c_opt > m:
                    c_opt = m
                    
                c_policy[t, z, i] = c_opt
                # Calculate value
                # V(t, z, m) = u(c) + beta * E[V(t+1, z', m')]
                next_wealths = R * (m - c_opt) + income
                exp_v_next = 0.0
                for zp in range(n_states):
                    v_next = np.interp(next_wealths[zp], resource_grid, V[t+1, zp, :])
                    exp_v_next += transition[z, zp] * v_next
                    
                V[t, z, i] = utility(c_opt) + beta * exp_v_next
                
    return c_policy
