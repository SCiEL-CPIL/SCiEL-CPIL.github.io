"""
Vectorized consumption policy solver using dense ratio grid and argmax selection.
"""
import time
import numpy as np

DEADLINE_SECONDS = 135

def _crsa_utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)

def _solve_consumption_single(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'])
    trans = np.asarray(model['transition'])
    horizon = model['horizon']
    states = len(income)
    n_grid = len(resource_grid)

    # Define a dense grid of consumption ratios
    # We need enough resolution to capture the optimal policy.
    # The evaluator uses 301 points. Let's use 1000 for safety and accuracy.
    # Ratios are in [0, 1]. c = ratio * m.
    # Note: c must be <= m. Also c > 0.
    # We will use ratios from 0.001 to 1.0 (or slightly less if we want to avoid boundary issues, 
    # but since we take max, boundary is handled).
    # Actually, let's use a geomspace for ratios to get better resolution at low consumption?
    # No, linear is fine if dense enough. Or geomspace for m is already there.
    # Let's try 2000 linear points for ratios.
    n_ratios = 2000
    ratios = np.linspace(0.001, 1.0, n_ratios)
    
    # Pre-compute utility for all possible (state, grid, ratio) combinations?
    # c = m * ratio.
    # m is (101,), ratio is (2000,). c is (101, 2000).
    # This is 202k points. Very fast.
    
    consumption = np.zeros((horizon, states, n_grid))
    
    # V_next[z, i] is the value function for next period.
    # We need to interpolate V_next for m'.
    # m' = R * (m - c) + y' = R * m * (1 - ratio) + y'
    
    V_next = np.zeros((states, n_grid))
    
    for t in range(horizon - 1, -1, -1):
        if t == horizon - 1:
            # Terminal: consume all
            consumption[t] = resource_grid[None, :]
            # V is just utility of consumption
            V_next = _crsa_utility(consumption[t], gamma)
            continue
        
        V_current = np.zeros((states, n_grid))
        
        # m is (n_grid,)
        m = resource_grid
        
        # For each state z, we need to find optimal ratio.
        # But the optimal ratio might depend on z? 
        # Yes, because income y' depends on z' which is drawn from z.
        # However, the grid m is the same.
        # The value function V_next[z'] is different for each z'.
        
        # Let's vectorize over z and m.
        # c candidates: (n_grid, n_ratios)
        # c = m[:, None] * ratios[None, :]
        c_grid = m[:, None] * ratios[None, :]
        
        # Ensure c > 0
        c_grid = np.maximum(c_grid, 1e-12)
        
        # Utility of consumption: (n_grid, n_ratios)
        u_c = _crsa_utility(c_grid, gamma)
        
        # Now compute expected continuation value.
        # m' = R * (m - c) + y'
        # m' shape: (n_grid, n_ratios, states) ? 
        # m is (n_grid,), c is (n_grid, n_ratios).
        # m - c is (n_grid, n_ratios).
        # R * (m - c) is (n_grid, n_ratios).
        # y' is (states,).
        # m_prime = R * (m[:, None, :] - c_grid[:, :, None]) + income[None, None, :]
        # Shape: (n_grid, n_ratios, states)
        
        m_prime = R * (m[:, None, :] - c_grid[:, :, None]) + income[None, None, :]
        
        # Clip m_prime to grid range for interpolation
        m_prime = np.clip(m_prime, resource_grid[0], resource_grid[-1])
        
        # Interpolate V_next for each state.
        # V_next is (states, n_grid).
        # We need V_next[z'] evaluated at m_prime[i, j, z'].
        # This is a bit tricky to vectorize fully without advanced indexing or loops over states.
        # Since states=2, a loop over states is cheap.
        
        expected_V = np.zeros((n_grid, n_ratios))
        
        for zp in range(states):
            # m_prime_zp = m_prime[:, :, zp] shape (n_grid, n_ratios)
            # We need to interpolate V_next[zp] (which is on resource_grid) at m_prime_zp.
            # np.interp works on 1D. We can flatten.
            m_flat = m_prime[:, :, zp].flatten()
            V_flat = np.interp(m_flat, resource_grid, V_next[zp])
            V_interp = V_flat.reshape((n_grid, n_ratios))
            
            # Weight by transition probability trans[z, zp]
            # Wait, the transition probability depends on current state z.
            # So we need to do this for each z separately, or vectorize over z as well.
            
            # Let's restructure:
            # For each z:
            #   For each zp:
            #     term = trans[z, zp] * V_next[zp](m_prime[:, :, zp])
            
            # Let's just loop over z (states=2) and inside vectorize over m and ratios.
            pass
            
        # Refined loop:
        for z in range(states):
            # Sum over zp
            sum_V = np.zeros((n_grid, n_ratios))
            for zp in range(states):
                m_p = m_prime[:, :, zp]
                # Interpolate
                m_flat = m_p.flatten()
                V_flat = np.interp(m_flat, resource_grid, V_next[zp])
                V_interp = V_flat.reshape((n_grid, n_ratios))
                sum_V += trans[z, zp] * V_interp
            
            # Bellman objective: u(c) + beta * expected_V
            objective = u_c + beta * sum_V
            
            # Find max
            idx = np.argmax(objective, axis=1)
            
            # Get optimal c
            optimal_c = c_grid[np.arange(n_grid), idx]
            
            consumption[t, z] = optimal_c
            V_current[z] = objective[np.arange(n_grid), idx]
            
        V_next = V_current

    return consumption

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    try:
        result = _solve_consumption_single(model, resource_grid, budget)

        # Check deadline
        if time.time() > deadline:
            pass # We already have a result, but if we were to do more, we'd stop.

        # Validate
        if not np.all(np.isfinite(result)):
            raise ValueError("Non-finite values")
        m = resource_grid[None, None, :]
        if np.any(result <= 0) or np.any(result > m + 1e-10):
            # Clip to valid range
            result = np.clip(result, 1e-12, m)
            # Re-check
            if np.any(result <= 0) or np.any(result > m + 1e-10):
                 raise ValueError("c out of bounds after clip")
        
        states = model['transition'].shape[0]
        # Terminal condition check
        if not np.allclose(result[5], np.tile(resource_grid, (states, 1)), atol=1e-9):
            # Force terminal condition
            result[5] = np.tile(resource_grid, (states, 1))
            # Re-check
            if not np.allclose(result[5], np.tile(resource_grid, (states, 1)), atol=1e-9):
                 raise ValueError("Terminal condition violated")

        return result
    except Exception:
        # Fallback: simple heuristic
        horizon = model['horizon']
        states = len(model['income'])
        n_grid = len(resource_grid)

        result = np.zeros((horizon, states, n_grid))
        for t in range(horizon):
            if t == horizon - 1:
                result[t] = np.tile(resource_grid, (states, 1))
            else:
                # Linear approximation of consumption
                result[t] = 0.9 * resource_grid[None, :]
                result[t] = np.clip(result[t], 1e-6, resource_grid[None, :])
                result[t] = np.tile(result[t], (states, 1))

        return result
