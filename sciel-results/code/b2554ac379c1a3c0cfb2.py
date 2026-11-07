import numpy as np
from functools import lru_cache

# Global cache for pre-computed policies
_POLICY_CACHE = {}

def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Pre-computes the optimal bid and ask depths for all time steps and inventory levels.
    Returns:
        opt_bid: np.array of shape (steps, 2*Q+1)
        opt_ask: np.array of shape (steps, 2*Q+1)
    """
    dt = T / steps
    N = steps
    q_range = np.arange(-Q, Q + 1)
    n_q = len(q_range)
    
    # V[t, i] where i is index in q_range
    # V has shape (N+1, n_q)
    V = np.zeros((N + 1, n_q))
    
    # Pre-compute grid for d
    # d must be > -3/k
    d_min = -3.0 / k + 1e-9
    # Upper bound for d: if d is very large, p is ~0.
    # A * exp(-k*d) * dt is small.
    # Let's choose d_max such that A*exp(-k*d_max)*dt < 1e-6
    # exp(-k*d_max) < 1e-6 / (A*dt)
    # -k*d_max < log(1e-6 / (A*dt))
    # d_max > -1/k * log(1e-6 / (A*dt))
    # If A*dt is small, log is large negative, d_max is large positive.
    # Let's just use a generous bound, e.g., 10/k or 20/k.
    d_max = 20.0 / k 
    d_grid = np.linspace(d_min, d_max, 200)
    
    # Pre-compute p(d) and d*p(d) for the grid?
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    # We need to optimize p(d)*(d + Delta)
    
    opt_bid = np.zeros((N, n_q))
    opt_ask = np.zeros((N, n_q))
    
    # Iterate backwards in time
    for t in range(N - 1, -1, -1):
        for i, q in enumerate(q_range):
            # Determine valid actions
            # If q == Q, no bid fill allowed -> d_bid = inf
            # If q == -Q, no ask fill allowed -> d_ask = inf
            
            # Get value differences from next step
            # q-1 index: i-1
            # q+1 index: i+1
            
            # Handle boundaries for V lookups
            # If q == Q, q+1 is out of bounds. But we don't buy, so we don't care about V(q+1) for bid?
            # Wait, if q=Q, we can only sell (ask).
            # If we sell, q becomes Q-1.
            # So we need V(Q-1).
            
            if q == Q:
                d_b_opt = np.inf
                p_b_opt = 0.0
                # For ask:
                # q' = q-1 = Q-1
                # V_next_q = V[t+1, i]
                # V_next_qm1 = V[t+1, i-1]
                # V_next_qp1 = V[t+1, i+1] (not used for ask if only one fill? No, ask fill reduces q)
                # Ask fill: q -> q-1.
                # So we need V[t+1, i] and V[t+1, i-1].
                # Delta_plus for ask?
                # The term is p_a * (d_a + (V(q-1) - V(q)))
                # Let Delta_ask = V[t+1, i-1] - V[t+1, i]
                delta_a = V[t + 1, i - 1] - V[t + 1, i]
                
                # Optimize d_a
                # Maximize p(d) * (d + delta_a)
                # Use grid
                p_grid = 1.0 - np.exp(-A * np.exp(-k * d_grid) * dt)
                rewards = p_grid * (d_grid + delta_a)
                idx = np.argmax(rewards)
                d_a_opt = d_grid[idx]
                p_a_opt = p_grid[idx]
                
                # Calculate V[t, i]
                # E[V_next] = (1-p_a) * V[t+1, i] + p_a * V[t+1, i-1]
                # (Bid prob is 0)
                exp_v = (1.0 - p_a_opt) * V[t + 1, i] + p_a_opt * V[t + 1, i - 1]
                cost = 0.5 * gamma * sigma**2 * q**2 * dt
                V[t, i] = -cost + p_a_opt * d_a_opt + exp_v
                
                opt_bid[t, i] = d_b_opt
                opt_ask[t, i] = d_a_opt
                
            elif q == -Q:
                d_a_opt = np.inf
                p_a_opt = 0.0
                # For bid:
                # Bid fill: q -> q+1 = -Q+1
                # Delta_bid = V[t+1, i+1] - V[t+1, i]
                delta_b = V[t + 1, i + 1] - V[t + 1, i]
                
                p_grid = 1.0 - np.exp(-A * np.exp(-k * d_grid) * dt)
                rewards = p_grid * (d_grid + delta_b)
                idx = np.argmax(rewards)
                d_b_opt = d_grid[idx]
                p_b_opt = p_grid[idx]
                
                exp_v = (1.0 - p_b_opt) * V[t + 1, i] + p_b_opt * V[t + 1, i + 1]
                cost = 0.5 * gamma * sigma**2 * q**2 * dt
                V[t, i] = -cost + p_b_opt * d_b_opt + exp_v
                
                opt_bid[t, i] = d_b_opt
                opt_ask[t, i] = d_a_opt
                
            else:
                # Interior state
                # Bid fill: q -> q-1. Delta_b = V[t+1, i-1] - V[t+1, i]
                # Ask fill: q -> q+1. Delta_a = V[t+1, i+1] - V[t+1, i]
                delta_b = V[t + 1, i - 1] - V[t + 1, i]
                delta_a = V[t + 1, i + 1] - V[t + 1, i]
                
                p_grid = 1.0 - np.exp(-A * np.exp(-k * d_grid) * dt)
                
                # Optimize Bid
                rewards_b = p_grid * (d_grid + delta_b)
                idx_b = np.argmax(rewards_b)
                d_b_opt = d_grid[idx_b]
                p_b_opt = p_grid[idx_b]
                
                # Optimize Ask
                rewards_a = p_grid * (d_grid + delta_a)
                idx_a = np.argmax(rewards_a)
                d_a_opt = d_grid[idx_a]
                p_a_opt = p_grid[idx_a]
                
                # Calculate V[t, i]
                # E[V_next] = (1-p_b)(1-p_a)V[i] + p_b(1-p_a)V[i-1] + (1-p_b)p_a V[i+1] + p_b p_a V[i]
                # = (1 - p_b - p_a + 2 p_b p_a) V[i] + p_b(1-p_a)V[i-1] + (1-p_b)p_a V[i+1]
                v_i = V[t + 1, i]
                v_im1 = V[t + 1, i - 1]
                v_ip1 = V[t + 1, i + 1]
                
                exp_v = (1.0 - p_b_opt - p_a_opt + 2.0 * p_b_opt * p_a_opt) * v_i + \
                        p_b_opt * (1.0 - p_a_opt) * v_im1 + \
                        (1.0 - p_b_opt) * p_a_opt * v_ip1
                        
                cost = 0.5 * gamma * sigma**2 * q**2 * dt
                # Reward from fills: p_b * d_b + p_a * d_a
                V[t, i] = -cost + p_b_opt * d_b_opt + p_a_opt * d_a_opt + exp_v
                
                opt_bid[t, i] = d_b_opt
                opt_ask[t, i] = d_a_opt
                
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    # Cache key
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _compute_policy(sigma, A, k, gamma, T, steps, Q)
        
    opt_bid, opt_ask = _POLICY_CACHE[key]
    
    dt = T / steps
    # Determine time step index
    # t = n * dt
    n = int(round(t / dt))
    # Clamp n to valid range [0, steps-1]
    n = np.clip(n, 0, steps - 1)
    
    # Map inventory to index
    # inventory is in [-Q, Q]
    # index = inventory + Q
    inv_idx = inventory + Q
    
    # Lookup depths
    # opt_bid has shape (steps, 2Q+1)
    # We want opt_bid[n, inv_idx]
    bid_depths = opt_bid[n, inv_idx]
    ask_depths = opt_ask[n, inv_idx]
    
    # Ensure output types are float64
    bid_depths = bid_depths.astype(np.float64)
    ask_depths = ask_depths.astype(np.float64)
    
    return bid_depths, ask_depths
