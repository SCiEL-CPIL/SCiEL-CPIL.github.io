# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for precomputed policies
_policy_cache = {}

def _solve_dp(params):
    """
    Solves the optimal policy using Dynamic Programming.
    Returns (opt_bid, opt_ask, dt)
    opt_bid: array of shape (steps, 2*Q+1)
    opt_ask: array of shape (steps, 2*Q+1)
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    dt = T / steps
    num_states = 2 * Q + 1
    
    # Grid for depths
    # Min depth: > -3/k
    min_depth = -3.0 / k + 1e-6
    
    # Max depth: Estimate where probability becomes negligible
    # P = 1 - exp(-A * exp(-k * b) * dt)
    # We want P < 1e-5 roughly
    # A * exp(-k * b) * dt ~ 1e-5
    # exp(-k * b) ~ 1e-5 / (A * dt)
    # -k * b ~ log(1e-5 / (A * dt))
    # b ~ -1/k * log(1e-5 / (A * dt))
    target_prob = 1e-5
    max_depth = -np.log(target_prob / (A * dt)) / k
    # Add a bit of buffer
    max_depth += 2.0 / k 
    
    # Ensure max_depth is not too small
    max_depth = max(max_depth, min_depth + 10.0 / k)
    
    grid_size = 300
    b_grid = np.linspace(min_depth, max_depth, grid_size)
    a_grid = np.linspace(min_depth, max_depth, grid_size)
    
    # Precompute probabilities for the grid
    # P_b depends only on b_grid
    # P_a depends only on a_grid
    # P = 1 - exp(-A * exp(-k * depth) * dt)
    
    exp_term_b = np.exp(-k * b_grid)
    P_b_1d = 1.0 - np.exp(-A * exp_term_b * dt)
    
    exp_term_a = np.exp(-k * a_grid)
    P_a_1d = 1.0 - np.exp(-A * exp_term_a * dt)
    
    # Precompute rewards per unit probability? No, reward is depth.
    # B_grid is b_grid, A_grid is a_grid.
    
    # Initialize DP tables
    opt_bid = np.full((steps, num_states), np.inf)
    opt_ask = np.full((steps, num_states), np.inf)
    
    V_next = np.zeros(num_states)
    
    # Precompute constants
    penalty_coeff = 0.5 * gamma * sigma * sigma * dt
    
    # Loop backwards in time
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(num_states)
        
        for q in range(-Q, Q + 1):
            q_idx = q + Q
            
            # Get next state values
            V_q = V_next[q_idx]
            
            # Determine valid transitions
            # If q == Q, no ask fill allowed (q+1 invalid)
            # If q == -Q, no bid fill allowed (q-1 invalid)
            
            if q == Q:
                # Only bid fills possible (q -> q-1)
                # Ask depth should be effectively infinity (P_a = 0)
                # We can just optimize over b, and set a to max_depth (or inf)
                # But strictly, if we quote a finite a, P_a > 0, which would lead to invalid state.
                # So we must set a such that P_a = 0.
                # In our grid, max_depth gives P_a ~ 1e-5. This is small but not 0.
                # To be safe, we can mask out any a that has P_a > 0.
                # Or simply, for boundary conditions, we can restrict the search.
                
                # Let's use the 2D grid but mask invalid regions.
                # Invalid: any a where P_a > 0 is technically invalid if it leads to q+1 > Q.
                # But q=Q is the max. So q+1 is out of bounds.
                # So we must ensure P_a = 0.
                # Since P_a > 0 for all finite a in our grid, we can't use the grid for a.
                # We must set a = np.inf.
                # Then P_a = 0.
                # The optimization reduces to 1D over b.
                
                # Calculate terms for 1D optimization over b
                # P_a = 0
                # P_bid_only = P_b
                # P_ask_only = 0
                # P_both = 0
                # P_stay = 1 - P_b
                # Reward = P_b * b
                # EV = P_b * V_next[q-1] + (1-P_b) * V_q
                
                # V_qm1 = V_next[q-1+Q]
                V_qm1 = V_next[q - 1 + Q]
                
                # Maximize: P_b * b + P_b * V_qm1 + (1 - P_b) * V_q
                # = P_b * (b + V_qm1) + V_q - P_b * V_q
                # = P_b * (b + V_qm1 - V_q) + V_q
                
                # Let F(b) = P_b * (b + V_qm1 - V_q)
                # We want to maximize F(b).
                
                # Vectorized over b_grid
                term = P_b_1d * (b_grid + V_qm1 - V_q)
                max_idx = np.argmax(term)
                best_b = b_grid[max_idx]
                best_a = np.inf # No quote on ask side effectively
                
                max_val = term[max_idx] + V_q
                # Subtract penalty
                V_curr[q_idx] = max_val - penalty_coeff * q * q
                
                opt_bid[n, q_idx] = best_b
                opt_ask[n, q_idx] = best_a
                
            elif q == -Q:
                # Only ask fills possible (q -> q+1)
                # Bid depth should be effectively infinity (P_b = 0)
                # Optimize over a.
                
                V_qp1 = V_next[q + 1 + Q]
                
                # Maximize: P_a * a + P_a * V_qp1 + (1 - P_a) * V_q
                # = P_a * (a + V_qp1 - V_q) + V_q
                
                term = P_a_1d * (a_grid + V_qp1 - V_q)
                max_idx = np.argmax(term)
                best_a = a_grid[max_idx]
                best_b = np.inf
                
                max_val = term[max_idx] + V_q
                V_curr[q_idx] = max_val - penalty_coeff * q * q
                
                opt_bid[n, q_idx] = best_b
                opt_ask[n, q_idx] = best_a
                
            else:
                # Both sides valid
                V_qm1 = V_next[q - 1 + Q]
                V_qp1 = V_next[q + 1 + Q]
                
                # 2D Grid Search
                # B shape (M, 1), A shape (1, M)
                B = b_grid[:, None]
                A_depth = a_grid[None, :]
                
                # P_b shape (M, 1), P_a shape (1, M)
                Pb = P_b_1d[:, None]
                Pa = P_a_1d[None, :]
                
                # Reward
                reward = Pb * B + Pa * A_depth
                
                # Probabilities
                P_bid_only = Pb * (1.0 - Pa)
                P_ask_only = (1.0 - Pb) * Pa
                P_both = Pb * Pa
                P_stay = 1.0 - P_bid_only - P_ask_only - P_both
                
                # Expected Value
                ev = P_bid_only * V_qm1 + P_ask_only * V_qp1 + P_stay * V_q
                
                total = reward + ev
                
                max_idx = np.unravel_index(np.argmax(total), total.shape)
                best_b = b_grid[max_idx[0]]
                best_a = a_grid[max_idx[1]]
                
                max_val = total[max_idx]
                V_curr[q_idx] = max_val - penalty_coeff * q * q
                
                opt_bid[n, q_idx] = best_b
                opt_ask[n, q_idx] = best_a
        
        V_next = V_curr
        
    return opt_bid, opt_ask, dt

def quote(t, inventory, mid, params):
    # Key for cache
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["T"], params["steps"], params["Q"])
    
    if key not in _policy_cache:
        # Check time
        start_time = time.time()
        opt_bid, opt_ask, dt = _solve_dp(params)
        elapsed = time.time() - start_time
        # If it took too long, we might have issues, but 300s is plenty.
        _policy_cache[key] = (opt_bid, opt_ask, dt)
    else:
        opt_bid, opt_ask, dt = _policy_cache[key]
    
    # Determine current step
    steps = params["steps"]
    n = int(t / dt)
    if n >= steps:
        n = steps - 1
        
    # Map inventory to index
    Q = params["Q"]
    q_indices = inventory + Q
    
    # Get optimal depths
    bid_depths = opt_bid[n, q_indices]
    ask_depths = opt_ask[n, q_indices]
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
