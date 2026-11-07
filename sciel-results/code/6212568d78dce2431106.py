import numpy as np
from functools import lru_cache

# Cache for DP solutions
_dp_cache = {}

def _solve_dp(params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps
    
    # State: n in 0..steps-1, q in -Q..Q
    # We store V[n, q+Q]
    V = np.zeros((steps + 1, 2 * Q + 1), dtype=np.float64)
    # opt_bid[n, q+Q], opt_ask[n, q+Q]
    opt_bid = np.zeros((steps, 2 * Q + 1), dtype=np.float64)
    opt_ask = np.zeros((steps, 2 * Q + 1), dtype=np.float64)
    
    # Grid for depth search
    # Depths are distances. Must be > -3/k.
    # We search in [0, 3/k]
    d_max = 3.0 / k
    n_grid = 20
    grid = np.linspace(0, d_max, n_grid)
    
    # Precompute exponentials for grid?
    # p(d) = 1 - exp(-A * exp(-k * d) * dt)
    
    for n in range(steps - 1, -1, -1):
        for q in range(-Q, Q + 1):
            idx_q = q + Q
            V_next_q = V[n + 1, idx_q]
            
            # Boundary conditions
            if q == Q:
                # No bid fill
                # Optimize a only
                # b is inf -> p_b = 0
                best_val = -np.inf
                best_a = 0.0
                for a in grid:
                    p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    # Reward: p_a * (a + V[n+1, q-1]) + (1-p_a) * V[n+1, q]
                    # Note: q-1 is valid since q=Q < Q+1
                    val = p_a * (a + V[n + 1, idx_q - 1]) + (1.0 - p_a) * V_next_q
                    val -= 0.5 * gamma * sigma**2 * q**2 * dt
                    if val > best_val:
                        best_val = val
                        best_a = a
                V[n, idx_q] = best_val
                opt_bid[n, idx_q] = np.inf
                opt_ask[n, idx_q] = best_a
            elif q == -Q:
                # No ask fill
                # a is inf -> p_a = 0
                best_val = -np.inf
                best_b = 0.0
                for b in grid:
                    p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    # Reward: p_b * (-b + V[n+1, q+1]) + (1-p_b) * V[n+1, q]
                    val = p_b * (-b + V[n + 1, idx_q + 1]) + (1.0 - p_b) * V_next_q
                    val -= 0.5 * gamma * sigma**2 * q**2 * dt
                    if val > best_val:
                        best_val = val
                        best_b = b
                V[n, idx_q] = best_val
                opt_bid[n, idx_q] = best_b
                opt_ask[n, idx_q] = np.inf
            else:
                # Interior
                best_val = -np.inf
                best_b = 0.0
                best_a = 0.0
                V_next_q_plus = V[n + 1, idx_q + 1]
                V_next_q_minus = V[n + 1, idx_q - 1]
                
                # Penalty is constant for given q
                penalty = 0.5 * gamma * sigma**2 * q**2 * dt
                
                for b in grid:
                    p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    # We can optimize a for this b? Or nested loop.
                    # Nested loop is 400 ops.
                    for a in grid:
                        p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                        # Expected immediate reward + continuation
                        # Fill bid: prob p_b, reward -b, next q+1
                        # Fill ask: prob p_a, reward a, next q-1
                        # No fill: prob (1-p_b)(1-p_a), next q
                        # Note: Fills are independent? "independently"
                        # So:
                        # E = p_b(1-p_a)(-b + V_q_plus) + p_a(1-p_b)(a + V_q_minus) + (1-p_b)(1-p_a)V_q
                        # Wait, can both fill? "bid hit with prob..., ask likewise, independently"
                        # Usually in discrete steps, we check if bid hits, if ask hits.
                        # If both hit, what happens?
                        # "A fill earns its depth and moves inventory one unit"
                        # If both hit, do we get -b + a and q stays q? Or q+1-1=q?
                        # Usually, if both hit, net inventory change is 0.
                        # Let's assume independent events.
                        # P(both) = p_b * p_a. Reward: -b + a. Next q.
                        # P(bid only) = p_b * (1-p_a). Reward: -b. Next q+1.
                        # P(ask only) = (1-p_b) * p_a. Reward: a. Next q-1.
                        # P(neither) = (1-p_b) * (1-p_a). Reward: 0. Next q.
                        
                        p_both = p_b * p_a
                        p_bid_only = p_b * (1.0 - p_a)
                        p_ask_only = (1.0 - p_b) * p_a
                        p_neither = (1.0 - p_b) * (1.0 - p_a)
                        
                        val = p_bid_only * (-b + V_next_q_plus) + \
                              p_ask_only * (a + V_next_q_minus) + \
                              p_both * (-b + a + V_next_q) + \
                              p_neither * V_next_q
                        val -= penalty
                        
                        if val > best_val:
                            best_val = val
                            best_b = b
                            best_a = a
                            
                V[n, idx_q] = best_val
                opt_bid[n, idx_q] = best_b
                opt_ask[n, idx_q] = best_a
                
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    # Create a cache key
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _dp_cache:
        _dp_cache[key] = _solve_dp(params)
        
    opt_bid, opt_ask = _dp_cache[key]
    
    # Determine step n
    dt = T / steps
    n = int(round(t / dt))
    # Clamp n to valid range
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    q = inventory
    # Map q to index: q + Q
    indices = q + Q
    
    # Retrieve optimal depths
    bid_depths = opt_bid[n, indices]
    ask_depths = opt_ask[n, indices]
    
    return bid_depths, ask_depths
