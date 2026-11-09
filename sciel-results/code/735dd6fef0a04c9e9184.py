"""Analytic HJB-based market making with inventory risk."""
import numpy as np
import time

# Module-level cache for precomputed values per parameter set
_cache = {}

def _get_tables(params):
    """Precompute lookup tables for the analytic policy given parameters."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # We'll use a continuous-time approximation via HJB.
    # The value function V(t, q) satisfies:
    # -V_t = gamma/2 * sigma^2 * q^2 + max_{b,a >= 0} [
    #     -A*exp(-k*b) * (b + E[V(t+dt, q-1)] - V(t,q))
    #     -A*exp(-k*a) * (a + E[V(t+dt, q+1)] - V(t,q))
    #     - sigma^2/2 * V_qq  (from Ito)
    # ]
    # 
    # In continuous time, the optimal bid depth b* and ask depth a* satisfy FOCs.
    # Let's discretize time backward and inventory to solve this.
    
    # Discretize: N_time = 1000 (matching steps), inventory from -Q to Q
    N_time = steps
    N_inv = 2 * Q + 1
    inv_grid = np.arange(-Q, Q + 1)
    
    # V[t, i] = value at time step t (0 = start, N_time-1 = last step), inventory index i
    V = np.zeros((N_time + 1, N_inv))
    
    # Boundary: at terminal time, V = 0 (no terminal charge)
    
    # Backward induction
    for t in range(N_time - 1, -1, -1):
        time_remaining = (N_time - t) * dt  # not directly needed in continuous approx
        for i in range(N_inv):
            q = inv_grid[i]
            
            # Inventory cost term
            inv_cost = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            # If at boundary, we can't quote on that side
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            # We need to optimize over bid_depth b >= 0 and ask_depth a >= 0
            # The expected continuation:
            # p_bid = 1 - exp(-A * exp(-k*b) * dt)
            # p_ask = 1 - exp(-A * exp(-k*a) * dt)
            # 
            # Expected change in V:
            # p_bid * (b + V[t+1, q-1] - V[t, q]) + p_ask * (a + V[t+1, q+1] - V[t, q])
            # But V[t,q] appears on both sides, so we need to solve for V[t,q].
            
            # Let's define:
            # Let v_next_minus = V[t+1, q-1] if q-1 in range else 0
            # Let v_next_plus = V[t+1, q+1] if q+1 in range else 0
            # Let v_next_same = V[t+1, q]
            
            # Actually, the mid also moves, but for the quote optimization at this step,
            # we condition on the mid being at its current value. The mid move affects
            # the next step's mid, but since we're doing backward induction on (t, q)
            # and the mid is exogenous (random walk), we can take expectation over mid moves.
            # However, the objective includes "depths earned" which are in price units,
            # and the mid move doesn't affect the depth earned (it's the distance from mid
            # at the time of quoting). So the mid move only affects future quotes.
            # 
            # For simplicity, and since the mid is a random walk with zero mean,
            # the expected future value given current mid is just V[t+1, q'] where q'
            # is the new inventory. The mid itself doesn't appear in the value function
            # explicitly because the objective is translation-invariant in mid (depths are
            # relative to mid).
            
            # So we can ignore the mid in the value function!
            
            # Now, for each (t, q), we need to find b*, a* that maximize:
            # -inv_cost + p_bid * (b + V[t+1, q-1] - V[t, q]) + p_ask * (a + V[t+1, q+1] - V[t, q])
            # - sigma^2/2 * V_qq  (discretized)
            
            # This is coupled because V[t,q] appears in the p_bid and p_ask terms.
            # Let's rearrange:
            # V[t,q] = -inv_cost - sigma^2/2 * V_qq + max_{b,a} [
            #     p_bid * (b + V[t+1, q-1]) + p_ask * (a + V[t+1, q+1])
            #     - (p_bid + p_ask) * V[t,q]
            # ]
            # 
            # So: V[t,q] * (1 + p_bid + p_ask) = -inv_cost - sigma^2/2 * V_qq + max_{b,a} [
            #     p_bid * (b + V[t+1, q-1]) + p_ask * (a + V[t+1, q+1])
            # ]
            
            # This is still coupled. Let's try a different approach: iterate to convergence.
            
            # Initialize V[t, q] with a guess
            v_guess = 0.0
            
            for iteration in range(20):
                # Optimize b and a given v_guess
                best_val = -np.inf
                best_b = 0.0
                best_a = 0.0
                
                # Grid search over b and a (coarse to fine)
                # b, a in [0, 20/k]
                max_depth = 20.0 / k
                n_grid = 50
                b_grid = np.linspace(0, max_depth, n_grid)
                a_grid = np.linspace(0, max_depth, n_grid)
                
                v_minus = V[t+1, i-1] if (can_bid and i > 0) else 0.0
                v_plus = V[t+1, i+1] if (can_ask and i < N_inv - 1) else 0.0
                
                # V_qq approximation
                v_qq = 0.0
                if i > 0 and i < N_inv - 1:
                    v_qq = (V[t+1, i-1] - 2*V[t+1, i] + V[t+1, i+1])
                
                for b in b_grid:
                    if not can_bid:
                        continue
                    p_bid = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    for a in a_grid:
                        if not can_ask:
                            continue
                        p_ask = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                        
                        val = (p_bid * (b + v_minus) + p_ask * (a + v_plus)
                               - (p_bid + p_ask) * v_guess
                               - inv_cost - 0.5 * sigma**2 * v_qq)
                        
                        if val > best_val:
                            best_val = val
                            best_b = b
                            best_a = a
                
                # Handle boundary cases
                if not can_bid:
                    best_b = np.inf
                if not can_ask:
                    best_a = np.inf
                
                # Update v_guess
                new_v = best_val
                if abs(new_v - v_guess) < 1e-10:
                    v_guess = new_v
                    break
                v_guess = 0.9 * v_guess + 0.1 * new_v
            
            V[t, i] = v_guess
    
    # Now extract optimal depths for each (t, q)
    # We need to redo the optimization to get the optimal b and a
    bid_depths_table = np.zeros((N_time, N_inv))
    ask_depths_table = np.zeros((N_time, N_inv))
    
    for t in range(N_time):
        for i in range(N_inv):
            q = inv_grid[i]
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            v_minus = V[t+1, i-1] if (can_bid and i > 0) else 0.0
            v_plus = V[t+1, i+1] if (can_ask and i < N_inv - 1) else 0.0
            v_qq = 0.0
            if i > 0 and i < N_inv - 1:
                v_qq = (V[t+1, i-1] - 2*V[t+1, i] + V[t+1, i+1])
            
            inv_cost = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            best_val = -np.inf
            best_b = 0.0
            best_a = 0.0
            
            max_depth = 20.0 / k
            n_grid = 100
            b_grid = np.linspace(0, max_depth, n_grid)
            a_grid = np.linspace(0, max_depth, n_grid)
            
            for b in b_grid:
                if not can_bid:
                    continue
                p_bid = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                for a in a_grid:
                    if not can_ask:
                        continue
                    p_ask = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    
                    val = (p_bid * (b + v_minus) + p_ask * (a + v_plus)
                           - (p_bid + p_ask) * V[t, i]
                           - inv_cost - 0.5 * sigma**2 * v_qq)
                    
                    if val > best_val:
                        best_val = val
                        best_b = b
                        best_a = a
            
            if not can_bid:
                best_b = np.inf
            if not can_ask:
                best_a = np.inf
            
            bid_depths_table[t, i] = best_b
            ask_depths_table[t, i] = best_a
    
    _cache[key] = (bid_depths_table, ask_depths_table, inv_grid)
    return _cache[key]


def quote(t, inventory, mid, params):
    """Quote using precomputed optimal policy tables."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Determine time step index
    dt = T / steps
    t_index = int(round(t / dt))
    t_index = np.clip(t_index, 0, steps - 1)
    
    # Get precomputed tables
    bid_table, ask_table, inv_grid = _get_tables(params)
    
    # Map inventory to index
    # inventory is in [-Q, Q], inv_grid is from -Q to Q
    inv_indices = np.clip(inventory + Q, 0, 2 * Q)
    
    # Get depths for each path
    bid_depths = bid_table[t_index, inv_indices]
    ask_depths = ask_table[t_index, inv_indices]
    
    # Ensure valid depths
    min_depth = -3.0 / k
    bid_depths = np.where(bid_depths < min_depth, np.inf, bid_depths)
    ask_depths = np.where(ask_depths < min_depth, np.inf, ask_depths)
    bid_depths = np.where(np.isnan(bid_depths), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), np.inf, ask_depths)
    
    return bid_depths, ask_depths
