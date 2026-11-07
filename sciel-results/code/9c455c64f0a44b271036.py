"""Optimal market making with inventory risk via precomputed HJB value function."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache for precomputed value functions
_value_cache = {}
_deadline = None
_start_time = None


def _compute_value_function(sigma, A, k, gamma, T, steps, Q):
    """
    Precompute the optimal value function using HJB discretization.
    Returns a 2D array V[t_idx, q] where t_idx is 0..steps and q is -Q..Q.
    """
    dt = T / steps
    n_states = 2 * Q + 1
    
    # V[t, q] = optimal expected reward from step t to end, starting with inventory q
    # We work backwards from t = steps (terminal) to t = 0
    # V[steps, q] = 0 (no terminal charge)
    
    # For each step, we choose bid_depth and ask_depth to maximize:
    #   immediate_reward + expected_future_value
    # 
    # The immediate reward is:
    #   - (gamma/2) * sigma^2 * q^2 * dt  (inventory holding cost)
    #   + bid_depth * P(bid_hit) + ask_depth * P(ask_hit)  (depth earned)
    #   - bid_depth * P(bid_hit) * 1  (inventory increases by 1)
    #   + ask_depth * P(ask_hit) * 1  (inventory decreases by 1)
    #
    # Wait, let me re-read the objective:
    # "depths earned minus sum over steps of (gamma / 2) sigma^2 q^2 dt"
    # So when a bid is hit, we earn bid_depth and inventory increases by 1.
    # When an ask is hit, we earn ask_depth and inventory decreases by 1.
    #
    # The probability of bid hit: 1 - exp(-A * exp(-k * bid_depth) * dt)
    # The probability of ask hit: 1 - exp(-A * exp(-k * ask_depth) * dt)
    #
    # These are independent.
    
    # Discretize depth space
    # Depth must be >= 0 and < 3/k (values below -3/k are invalid, so depth must be < 3/k)
    # Actually, the constraint says values below -3/k make result invalid.
    # So bid_depth must be > -3/k. Since depth is a distance, it should be non-negative.
    # Let's use depth in [0, 3/k)
    
    max_depth = 3.0 / k
    n_depths = 200
    depths = np.linspace(0.0, max_depth * 0.999, n_depths)
    
    # Precompute hit probabilities for each depth
    # P_hit(d) = 1 - exp(-A * exp(-k * d) * dt)
    lambda_vals = A * np.exp(-k * depths) * dt
    p_hits = 1.0 - np.exp(-lambda_vals)
    
    # V[t, q] for t = steps down to 0
    # q index: 0 corresponds to q = -Q, n_states-1 corresponds to q = Q
    V = np.zeros((steps + 1, n_states))
    
    for t in range(steps - 1, -1, -1):
        for qi in range(n_states):
            q = qi - Q  # actual inventory
            
            # If at boundary, we can only quote one side
            if q == Q:
                # No bid fill possible, only ask
                # Choose ask_depth to maximize:
                #   - (gamma/2)*sigma^2*q^2*dt + ask_depth * p_ask + V[t+1, q-1] * p_ask + V[t+1, q] * (1-p_ask)
                # Note: if ask hits, inventory goes from Q to Q-1
                best_val = -np.inf
                for di in range(n_depths):
                    d = depths[di]
                    p = p_hits[di]
                    val = -(gamma / 2.0) * sigma**2 * q**2 * dt
                    val += d * p
                    val += p * V[t + 1, qi - 1]
                    val += (1 - p) * V[t + 1, qi]
                    if val > best_val:
                        best_val = val
                V[t, qi] = best_val
            elif q == -Q:
                # No ask fill possible, only bid
                best_val = -np.inf
                for di in range(n_depths):
                    d = depths[di]
                    p = p_hits[di]
                    val = -(gamma / 2.0) * sigma**2 * q**2 * dt
                    val += d * p
                    val += p * V[t + 1, qi + 1]
                    val += (1 - p) * V[t + 1, qi]
                    if val > best_val:
                        best_val = val
                V[t, qi] = best_val
            else:
                # Both sides possible
                # Choose bid_depth and ask_depth
                # This is a 2D optimization, which is expensive
                # We can use a grid search over bid_depth and ask_depth
                best_val = -np.inf
                # To speed up, we can use a coarser grid first then refine
                for di in range(n_depths):
                    d_bid = depths[di]
                    p_bid = p_hits[di]
                    for dj in range(n_depths):
                        d_ask = depths[dj]
                        p_ask = p_hits[dj]
                        
                        val = -(gamma / 2.0) * sigma**2 * q**2 * dt
                        # Expected immediate reward
                        val += d_bid * p_bid * (1 - p_ask)
                        val += d_ask * p_ask * (1 - p_bid)
                        # If both hit: earn both depths, inventory unchanged (bid +1, ask -1)
                        val += (d_bid + d_ask) * p_bid * p_ask
                        # Future value
                        # P(both hit) = p_bid * p_ask: q stays same
                        # P(bid only) = p_bid * (1-p_ask): q -> q+1
                        # P(ask only) = (1-p_bid) * p_ask: q -> q-1
                        # P(neither) = (1-p_bid) * (1-p_ask): q stays same
                        val += p_bid * p_ask * V[t + 1, qi]
                        val += p_bid * (1 - p_ask) * V[t + 1, qi + 1]
                        val += (1 - p_bid) * p_ask * V[t + 1, qi - 1]
                        val += (1 - p_bid) * (1 - p_ask) * V[t + 1, qi]
                        
                        if val > best_val:
                            best_val = val
                V[t, qi] = best_val
    
    return V


def quote(t, inventory, mid, params):
    global _deadline, _start_time
    
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 280.0  # 280s deadline with 50s margin
    
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    dt = T / steps
    t_idx = int(round(t / dt))
    if t_idx >= steps:
        t_idx = steps - 1
    
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    
    if cache_key not in _value_cache:
        _value_cache[cache_key] = _compute_value_function(sigma, A, k, gamma, T, steps, Q)
    
    V = _value_cache[cache_key]
    
    # Now, for each path, determine optimal depths
    # We need to find the optimal bid_depth and ask_depth for each inventory level
    # This requires re-doing the optimization, but we can vectorize it
    
    n_paths = len(inventory)
    bid_depths = np.full(n_paths, np.inf)
    ask_depths = np.full(n_paths, np.inf)
    
    # Discretize depth space (same as in _compute_value_function)
    max_depth = 3.0 / k
    n_depths = 200
    depths = np.linspace(0.0, max_depth * 0.999, n_depths)
    lambda_vals = A * np.exp(-k * depths) * dt
    p_hits = 1.0 - np.exp(-lambda_vals)
    
    # For each path, find optimal depths
    for i in range(n_paths):
        q = inventory[i]
        qi = q + Q
        
        if q == Q:
            # Only ask side
            best_val = -np.inf
            best_d = 0.0
            for di in range(n_depths):
                d = depths[di]
                p = p_hits[di]
                val = -(gamma / 2.0) * sigma**2 * q**2 * dt
                val += d * p
                val += p * V[t_idx + 1, qi - 1]
                val += (1 - p) * V[t_idx + 1, qi]
                if val > best_val:
                    best_val = val
                    best_d = d
            ask_depths[i] = best_d
        elif q == -Q:
            # Only bid side
            best_val = -np.inf
            best_d = 0.0
            for di in range(n_depths):
                d = depths[di]
                p = p_hits[di]
                val = -(gamma / 2.0) * sigma**2 * q**2 * dt
                val += d * p
                val += p * V[t_idx + 1, qi + 1]
                val += (1 - p) * V[t_idx + 1, qi]
                if val > best_val:
                    best_val = val
                    best_d = d
            bid_depths[i] = best_d
        else:
            # Both sides
            best_val = -np.inf
            best_db = 0.0
            best_da = 0.0
            for di in range(n_depths):
                d_bid = depths[di]
                p_bid = p_hits[di]
                for dj in range(n_depths):
                    d_ask = depths[dj]
                    p_ask = p_hits[dj]
                    
                    val = -(gamma / 2.0) * sigma**2 * q**2 * dt
                    val += d_bid * p_bid * (1 - p_ask)
                    val += d_ask * p_ask * (1 - p_bid)
                    val += (d_bid + d_ask) * p_bid * p_ask
                    val += p_bid * p_ask * V[t_idx + 1, qi]
                    val += p_bid * (1 - p_ask) * V[t_idx + 1, qi + 1]
                    val += (1 - p_bid) * p_ask * V[t_idx + 1, qi - 1]
                    val += (1 - p_bid) * (1 - p_ask) * V[t_idx + 1, qi]
                    
                    if val > best_val:
                        best_val = val
                        best_db = d_bid
                        best_da = d_ask
            bid_depths[i] = best_db
            ask_depths[i] = best_da
    
    # Check deadline
    if time.time() > _deadline:
        # Return simple fallback
        depth = np.full(n_paths, 1.0 / k)
        return depth, depth.copy()
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
