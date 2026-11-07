"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    """
    Solve finite-horizon consumption-saving problem.
    
    Parameters:
    - model: dict with beta, gamma, R, income, transition, horizon
    - resource_grid: np.array of grid points
    - budget: int, number of consumption search points per grid point
    
    Returns:
    - consumption: array of shape (horizon, n_income_states, len(resource_grid))
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c**(1 - gamma) / (1 - gamma)
    
    def util_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c**(-gamma)
    
    def util_deriv2(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return -1.0 / c**2
        else:
            return -gamma * c**(-gamma - 1)
    
    # Initialize consumption policy
    # consumption[t, z, i] = optimal consumption at time t, state z, grid index i
    consumption = np.zeros((horizon, n_states, n_grid))
    
    # Terminal condition: at t=horizon-1, consume everything
    for z in range(n_states):
        consumption[horizon - 1, z, :] = resource_grid
    
    # Backward induction
    # For t from horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5.0:
            # Fallback: use simple heuristic
            for z in range(n_states):
                consumption[t, z, :] = np.maximum(resource_grid - income[z] / R, 0.001)
            break
        
        # For each state z
        for z in range(n_states):
            # For each grid point m
            for i in range(n_grid):
                m = resource_grid[i]
                
                # Search over consumption using Newton's method on Euler equation
                # Euler equation: u'(c) = beta * R * sum_{z'} P(z'|z) * u'(c')
                # where c' = R*(m-c) + income[z']
                
                # Initial guess: proportional rule
                c_guess = min(0.5 * m + 0.5 * income[z] / (1 + R), m)
                c_guess = max(c_guess, 1e-12)
                
                c_opt = c_guess
                converged = False
                
                # Newton iterations
                for newton_iter in range(20):
                    if time.time() > deadline - 2.0:
                        break
                    
                    # Compute continuation value
                    c_remaining = R * (m - c_opt)
                    # Check feasibility
                    if c_remaining < -1e-10:
                        c_opt = max(m * 0.999, 1e-12)
                        continue
                    
                    # Expected marginal utility of next period
                    expected_mu_next = 0.0
                    for zp in range(n_states):
                        prob = transition[z, zp]
                        if prob < 1e-15:
                            continue
                        c_next = c_remaining + income[zp]
                        c_next = max(c_next, 1e-12)
                        # For terminal period, c' = m' (consume all)
                        if t + 1 == horizon - 1:
                            # At terminal, consumption = resources, so c' = m'
                            expected_mu_next += prob * util_deriv(c_next)
                        else:
                            # Use policy from next period
                            c_next_policy = consumption[t + 1, zp, :]
                            # Interpolate to find c' for resource m'
                            m_prime = c_remaining + income[zp]
                            # Find c' using interpolation
                            if m_prime <= resource_grid[0]:
                                c_prime = c_next_policy[0]
                            elif m_prime >= resource_grid[-1]:
                                c_prime = c_next_policy[-1]
                            else:
                                c_prime = np.interp(m_prime, resource_grid, c_next_policy)
                            c_prime = max(c_prime, 1e-12)
                            expected_mu_next += prob * util_deriv(c_prime)
                    
                    # Euler equation residual
                    resid = util_deriv(c_opt) - beta * R * expected_mu_next
                    deriv = util_deriv2(c_opt)
                    
                    if abs(deriv) < 1e-30:
                        break
                    
                    c_new = c_opt - resid / deriv
                    c_new = max(c_new, 1e-12)
                    c_new = min(c_new, m)
                    
                    if abs(c_new - c_opt) < 1e-10:
                        c_opt = c_new
                        converged = True
                        break
                    
                    c_opt = c_new
                
                consumption[t, z, i] = c_opt
        
        # Check time
        if time.time() > deadline - 10.0:
            # Use heuristic for remaining periods
            for tt in range(t, -1, -1):
                for z in range(n_states):
                    # Simple heuristic: save enough to maintain income
                    save_target = income[z] / R
                    consumption[tt, z, :] = np.maximum(resource_grid - save_target, 0.001)
            break
    
    # Enforce constraints: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_states):
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], resource_grid + 1e-10)
    
    # Terminal condition: c[5] = grid
    consumption[horizon - 1, :, :] = resource_grid
    
    return consumption
# EVOLVE-BLOCK-END
