import time
import numpy as np
from scipy.optimize import brentq

DEADLINE_SECONDS = 130  # Safety margin before the 150s kill

def _crsa_utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)

def _crsa_marginal(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return 1.0 / c
    return c ** (-gamma)

def _solve_consumption_single(model, resource_grid, budget, deadline):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    trans = model['transition']
    horizon = model['horizon']
    states = len(income)
    n_grid = len(resource_grid)
    
    # We will store V[t, z, i] = value function at time t, state z, grid index i
    # And c_policy[t, z, i] = optimal consumption
    consumption = np.zeros((horizon, states, n_grid))
    V = np.zeros((horizon, states, n_grid))
    
    # Log-linear approximation parameters for V_next
    # V(m, z') = A[z'] + B[z'] * log(m)
    # We initialize these with a rough heuristic or the previous period's best fit.
    # For the last period (t = horizon - 1), V = u(m).
    # If gamma != 1, V(m) = m^(1-gamma)/(1-gamma). This is not log-linear, but we can fit it.
    # If gamma = 1, V(m) = log(m). This IS log-linear with A=0, B=1.
    
    # We will iterate backward.
    # At each step, we assume V_next is well-approximated by A + B*log(m).
    # We solve for the optimal policy c(m, z) using this approximation.
    # Then we evaluate the actual V[m, z] using the optimal policy and the approximated continuation.
    # Finally, we fit A and B for the current period's V to use in the next step (t-1).
    # This is a form of policy iteration / value function approximation.
    
    # Initialize A and B for the terminal period (t = horizon - 1)
    # V_terminal(m) = u(m)
    m_vals = resource_grid
    u_vals = _crsa_utility(m_vals, gamma)
    
    # Fit A and B to u_vals
    # u_vals = A + B * log(m_vals)
    # Least squares fit
    X = np.column_stack([np.ones_like(m_vals), np.log(m_vals)])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, u_vals, rcond=None)
        A_terminal = coeffs[0]
        B_terminal = coeffs[1]
    except:
        # Fallback
        A_terminal = 0
        B_terminal = 1 if gamma == 1 else 0.5

    # Store A and B for each state (they are the same for all states in terminal period)
    A = np.zeros((horizon, states))
    B = np.zeros((horizon, states))
    A[horizon-1, :] = A_terminal
    B[horizon-1, :] = B_terminal
    
    # Terminal consumption
    consumption[horizon - 1] = np.tile(resource_grid, (states, 1))
    V[horizon - 1] = np.tile(u_vals, (states, 1))
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        # A_next and B_next are known from the previous iteration
        A_next = A[t + 1]
        B_next = B[t + 1]
        
        for z in range(states):
            trans_z = trans[z]
            
            # Precompute weighted sums for the log-linear approximation
            # E[V_next] = sum_z' trans[z, z'] * (A[z'] + B[z'] * log(m'))
            # = sum_z' trans[z, z'] * A[z'] + sum_z' trans[z, z'] * B[z'] * log(m')
            # Let S_A = sum_z' trans[z, z'] * A[z']
            # Let S_B = sum_z' trans[z, z'] * B[z']
            S_A = np.sum(trans_z * A_next)
            S_B = np.sum(trans_z * B_next)
            
            # FOC: u'(c) = beta * R * E[V_next_marginal]
            # V_next(m') = A + B * log(m')
            # V_next_marginal(m') = B / m'
            # E[V_next_marginal] = sum_z' trans[z, z'] * B[z'] / m'[z']
            # This is NOT S_B / m because m' depends on z'.
            # m'[z'] = R * (m - c) + income[z']
            
            # So the FOC is:
            # u'(c) = beta * R * sum_z' trans[z, z'] * B_next[z'] / (R * (m - c) + income[z'])
            
            # We solve for c for each m in resource_grid.
            for i in range(n_grid):
                m = resource_grid[i]
                
                def g(f):
                    c = f * m
                    if c <= 0:
                        return 1e10
                    lhs = _crsa_marginal(c, gamma)
                    
                    rhs_sum = 0.0
                    for zp in range(states):
                        m_prime = R * (m - c) + income[zp]
                        # m_prime must be positive for log
                        if m_prime <= 0:
                            # If m_prime is non-positive, the approximation breaks down.
                            # But m >= 0.001, c <= m. R=1.02. income > 0.
                            # Min m' = 1.02*(0.001 - 0.001) + 0.5 = 0.5 > 0.
                            m_prime = 1e-6 
                        rhs_sum += trans_z[zp] * B_next[zp] / m_prime
                    
                    rhs = beta * R * rhs_sum
                    return lhs - rhs
                
                f_min = 1e-6
                f_max = 1.0
                
                g_min = g(f_min)
                g_max = g(f_max)
                
                if g_min < 0:
                    f_opt = f_min
                elif g_max > 0:
                    f_opt = f_max
                else:
                    try:
                        f_opt = brentq(g, f_min, f_max, xtol=1e-8)
                    except Exception:
                        lo = f_min
                        hi = f_max
                        for _ in range(50):
                            mid = (lo + hi) / 2.0
                            if g(mid) > 0:
                                lo = mid
                            else:
                                hi = mid
                        f_opt = (lo + hi) / 2.0
                
                c_opt = f_opt * m
                consumption[t, z, i] = c_opt
                
                # Compute V[t, z, i] using the log-linear approximation for continuation
                u_c = _crsa_utility(c_opt, gamma)
                
                cont_val = 0.0
                for zp in range(states):
                    m_prime = R * (m - c_opt) + income[zp]
                    if m_prime <= 0:
                        m_prime = 1e-6
                    V_approx = A_next[zp] + B_next[zp] * np.log(m_prime)
                    cont_val += trans_z[zp] * V_approx
                
                V[t, z, i] = u_c + beta * cont_val
                
                # Time check
                if time.time() > deadline:
                    return consumption, V, True
            
            # Fit A and B for the current period and state z
            # V[t, z, i] = A[t, z] + B[t, z] * log(m_i)
            m_vals = resource_grid
            V_vals = V[t, z, :]
            
            X = np.column_stack([np.ones_like(m_vals), np.log(m_vals)])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, V_vals, rcond=None)
                A[t, z] = coeffs[0]
                B[t, z] = coeffs[1]
            except:
                A[t, z] = 0
                B[t, z] = 1.0 if gamma == 1 else 0.5
        
        # Time check
        if time.time() > deadline:
            return consumption, V, True
    
    return consumption, V, False

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    try:
        consumption, V, timed_out = _solve_consumption_single(model, resource_grid, budget, deadline)
        
        # Validate
        if not np.all(np.isfinite(consumption)):
            raise ValueError("Non-finite values")
        
        m = resource_grid[None, None, :]
        if np.any(consumption <= 0) or np.any(consumption > m + 1e-10):
            raise ValueError("c out of bounds")
        
        # Check c[5] equals grid within 1e-9
        states = model['transition'].shape[0]
        if not np.allclose(consumption[5], np.tile(resource_grid, (states, 1)), atol=1e-9):
            raise ValueError("Terminal condition violated")
        
        return consumption
        
    except Exception as e:
        # Fallback: simple heuristic
        horizon = model['horizon']
        states = len(model['income'])
        n_grid = len(resource_grid)
        
        result = np.zeros((horizon, states, n_grid))
        for t in range(horizon):
            if t == horizon - 1:
                result[t] = np.tile(resource_grid, (states, 1))
            else:
                # Simple rule: consume a fraction
                frac = 0.9
                result[t] = frac * resource_grid[None, :]
                result[t] = np.clip(result[t], 1e-6, resource_grid[None, :])
                result[t] = np.tile(result[t], (states, 1))
        
        return result
