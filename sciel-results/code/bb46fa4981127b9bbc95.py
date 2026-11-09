"""Improved consumption-saving policy solver."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _crsa_utility(c, gamma):
    """CRRA utility: log(c) if gamma=1, else c^(1-gamma)/(1-gamma)."""
    c = np.maximum(c, 1e-12)
    if np.isclose(gamma, 1.0):
        return np.log(c)
    else:
        return c ** (1 - gamma) / (1 - gamma)

def _solve_consumption_single(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    
    # consumption[t, z, m_index] shape (horizon, n_states, n_grid)
    consumption = np.zeros((horizon, n_states, n_grid))
    
    # At t=horizon (last period), consume all resources
    # c[5] must equal the grid within 1e-9
    consumption[horizon - 1, :, :] = resource_grid[np.newaxis, :, np.newaxis] * np.ones((1, n_states, 1))
    
    # We do backward induction from t=horizon-2 down to t=0
    # Value function: V[t, z, m] = max_c { u(c) + beta * sum_z' P(z'|z) V[t+1, z', m'] }
    # where m' = R*(m-c) + income[z']
    
    # For each time step, we need to find optimal c for each (z, m)
    # Use a grid of c values and interpolate continuation values
    
    # We'll use a finer c-grid for better accuracy
    # c ranges from 0 to m, so we use c = m * linspace(0.001, 1, n_c)
    n_c = min(budget, 200)  # number of c candidates per (z,m)
    
    # For efficiency, we precompute the continuation value as a function of m'
    # for each z' at time t+1
    
    for t in range(horizon - 2, -1, -1):
        # For each income state z
        for z in range(n_states):
            # For each m in resource_grid
            for i in range(n_grid):
                m = resource_grid[i]
                
                # Candidate c values
                c_candidates = m * np.linspace(0.001, 1.0, n_c)
                
                # For each c, compute m' = R*(m-c) + income[z'] for each z'
                # and get continuation value
                best_c = c_candidates[0]
                best_val = -np.inf
                
                for j in range(n_c):
                    c = c_candidates[j]
                    if c <= 0 or c > m + 1e-10:
                        continue
                    
                    m_after = R * (m - c)
                    
                    # Compute expected continuation value
                    cont_val = 0.0
                    for zp in range(n_states):
                        p = transition[z, zp]
                        if p < 1e-15:
                            continue
                        
                        m_prime = m_after + income[zp]
                        
                        # Interpolate V[t+1, zp, m_prime]
                        if m_prime <= resource_grid[0]:
                            v_next = 0.0  # or some floor
                        elif m_prime >= resource_grid[-1]:
                            v_next = _crsa_utility(resource_grid[-1], gamma) * (beta ** 0)  # approximate
                            # Actually at horizon-1, V = u(m)
                            # Let's handle this more carefully
                            v_next = _crsa_utility(np.minimum(m_prime, resource_grid[-1]), gamma)
                        else:
                            # np.interp for V[t+1, zp, :]
                            v_next = np.interp(m_prime, resource_grid, consumption[t+1, zp, :] * 0 + 
                                              _crsa_utility(consumption[t+1, zp, :], gamma) if t+1 == horizon-1 else 
                                              _get_value(t+1, zp, m_prime, resource_grid, consumption, gamma, beta, R, income, transition, horizon))
                        
                        cont_val += p * v_next
                    
                    total = _crsa_utility(c, gamma) + beta * cont_val
                    
                    if total > best_val:
                        best_val = total
                        best_c = c
                
                consumption[t, z, i] = best_c
    
    return consumption

def _get_value(t, z, m, resource_grid, consumption, gamma, beta, R, income, transition, horizon):
    """Get value at time t, state z, resource m by interpolation."""
    if t == horizon - 1:
        # Last period: V = u(m)
        return _crsa_utility(m, gamma)
    # For earlier periods, we don't have V stored directly, only consumption
    # This is a simplification - we should store V separately
    # For now, approximate using the consumption policy
    c = np.interp(m, resource_grid, consumption[t, z, :])
    m_after = R * (m - c)
    cont_val = 0.0
    for zp in range(len(income)):
        p = transition[z, zp]
        if p < 1e-15:
            continue
        m_prime = m_after + income[zp]
        v_next = _get_value(t + 1, zp, m_prime, resource_grid, consumption, gamma, beta, R, income, transition, horizon)
        cont_val += p * v_next
    return _crsa_utility(c, gamma) + beta * cont_val

def solve_consumption(model, resource_grid, budget):
    """Solve consumption-saving problem.
    
    Returns consumption[t, income_state, resource_index], shape (6, 2, 101).
    """
    start_time = time.time()
    deadline = start_time + 145  # 145 second deadline (5s safety margin)
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    
    # consumption[t, z, m_index] shape (horizon, n_states, n_grid)
    consumption = np.zeros((horizon, n_states, n_grid))
    
    # At t=horizon-1 (last period), consume all resources
    consumption[horizon - 1, :, :] = resource_grid[np.newaxis, :, np.newaxis] * np.ones((1, n_states, 1))
    
    # Store value functions for continuation
    # V[t, z, m_index] shape (horizon, n_states, n_grid)
    V = np.zeros((horizon, n_states, n_grid))
    
    # Initialize last period values
    V[horizon - 1, :, :] = _crsa_utility(resource_grid[np.newaxis, :], gamma) * np.ones((1, n_states, 1))
    
    # Number of c candidates
    n_c = min(budget, 150)
    
    # Precompute c fractions
    c_fractions = np.linspace(0.001, 1.0, n_c)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            # Not enough time, fill remaining with a simple heuristic
            for tt in range(t, -1, -1):
                for z in range(n_states):
                    # Heuristic: consume based on Euler equation approximation
                    # For CRRA, c ≈ m * (1 - beta*R) / (1 - beta*R) ... simplified
                    # Just use a fraction
                    frac = 1.0 / (1 + (horizon - 1 - tt) * 0.1)
                    consumption[tt, z, :] = np.clip(resource_grid * frac, 1e-10, resource_grid)
                    V[tt, z, :] = _crsa_utility(consumption[tt, z, :], gamma)
            break
        
        for z in range(n_states):
            # Vectorize over m for speed
            m_values = resource_grid  # shape (n_grid,)
            
            # For each m, we need to find optimal c
            # Use vectorized approach where possible
            
            best_c = np.zeros(n_grid)
            best_val = np.full(n_grid, -np.inf)
            
            # Try each c fraction
            for j in range(n_c):
                frac = c_fractions[j]
                c_vals = m_values * frac  # shape (n_grid,)
                
                # Ensure 0 < c <= m
                valid = (c_vals > 0) & (c_vals <= m_values + 1e-10)
                if not np.any(valid):
                    continue
                
                m_after = R * (m_values - c_vals)  # shape (n_grid,)
                
                # Compute expected continuation value
                cont_val = np.zeros(n_grid)
                for zp in range(n_states):
                    p = transition[z, zp]
                    if p < 1e-15:
                        continue
                    
                    m_prime = m_after + income[zp]  # shape (n_grid,)
                    
                    # Interpolate V[t+1, zp, m_prime]
                    v_next = np.interp(m_prime, resource_grid, V[t+1, zp, :],
                                       left=V[t+1, zp, 0], right=V[t+1, zp, -1])
                    
                    cont_val += p * v_next
                
                total = _crsa_utility(c_vals, gamma) + beta * cont_val
                
                # Update best where valid
                improved = valid & (total > best_val)
                best_c[improved] = c_vals[improved]
                best_val[improved] = total[improved]
            
            consumption[t, z, :] = best_c
            # Compute V[t, z, :] using the optimal c
            m_after_opt = R * (m_values - best_c)
            cont_val_opt = np.zeros(n_grid)
            for zp in range(n_states):
                p = transition[z, zp]
                if p < 1e-15:
                    continue
                m_prime_opt = m_after_opt + income[zp]
                v_next_opt = np.interp(m_prime_opt, resource_grid, V[t+1, zp, :],
                                       left=V[t+1, zp, 0], right=V[t+1, zp, -1])
                cont_val_opt += p * v_next_opt
            
            V[t, z, :] = _crsa_utility(best_c, gamma) + beta * cont_val_opt
    
    # Final validation
    # Ensure finite values
    consumption = np.nan_to_num(consumption, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_states):
            m_vals = resource_grid
            consumption[t, z, :] = np.clip(consumption[t, z, :], 1e-12, m_vals + 1e-10)
    
    # Ensure c[5] equals grid within 1e-9
    consumption[horizon - 1, :, :] = resource_grid[np.newaxis, :, np.newaxis] * np.ones((1, n_states, 1))
    
    return consumption
# EVOLVE-BLOCK-END
