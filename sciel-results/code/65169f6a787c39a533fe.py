"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-computation cache for the optimal policy
# We solve the discrete optimal control problem using Dynamic Programming.
# State: (n, q) where n is time step (0..999), q is inventory (-Q..Q).
# Action: (bid_depth, ask_depth).
# Since we can choose continuous depths, we discretize the depth space.
# The optimal depth will likely be small. We search a grid of depths.

_POLICY_CACHE = {}

def _compute_policy(sigma, A, k, gamma, T, steps, Q, depth_grid):
    """
    Computes the optimal bid and ask depths for each (time, inventory) state.
    Returns a dictionary mapping (n, q) to (opt_bid, opt_ask).
    """
    dt = T / steps
    
    # Depth grid: from very small (aggressive) to large (conservative)
    # We expect optimal depths to be around 1/k.
    # Let's use a fine grid around the expected range.
    # Min depth: 0.01 (very aggressive)
    # Max depth: 5.0 (very conservative)
    # Number of points: 50
    depths = depth_grid
    n_depths = len(depths)
    
    # State space
    # q ranges from -Q to Q
    # We create a 2D array V[n, q_idx]
    # q_idx = q + Q
    
    # Initialize V at T (n=steps) to 0 (no terminal charge)
    V = np.zeros((steps + 1, 2 * Q + 1))
    
    # Optimal actions
    opt_bid = np.zeros((steps, 2 * Q + 1))
    opt_ask = np.zeros((steps, 2 * Q + 1))
    
    # Pre-compute fill probabilities for each depth
    # p_fill(d) = 1 - exp(-A * exp(-k * d) * dt)
    # Note: if d is very large, exp(-k*d) is 0, p=0.
    # If d is small, exp(-k*d) is ~1, p ~ 1 - exp(-A*dt).
    
    # To speed up, we can precompute the cost for each depth choice given current V.
    # But V changes at each step.
    
    # We iterate backwards in time
    # n goes from steps-1 down to 0
    
    # Optimization: 
    # For a given state (n, q), we choose (b, a) to minimize:
    # Cost = gamma/2 * sigma^2 * q^2 * dt  (Inventory cost for this step)
    #      + p_bid(b) * ( -b + V[n+1, q+1] )  # If bid hit: earn b, inventory increases by 1 (becomes q+1)
    #        * (1 if q < Q else 0)             # No bid fill at Q
    #      + p_ask(a) * ( -a + V[n+1, q-1] )  # If ask hit: earn a (but it's a sale, so we "earn" the price difference? 
    #                                          # Wait, objective is "depths earned". 
    #                                          # If I sell at ask_depth a above mid, I earn a.
    #                                          # Inventory decreases by 1 (becomes q-1).
    #        * (1 if q > -Q else 0)            # No ask fill at -Q
    #      + (1 - p_bid(b)) * (1 - p_ask(a)) * V[n+1, q]  # No fill
    #      + p_bid(b) * (1 - p_ask(a)) * V[n+1, q+1]      # Only bid fill
    #      + (1 - p_bid(b)) * p_ask(a) * V[n+1, q-1]      # Only ask fill
    #      + p_bid(b) * p_ask(a) * V[n+1, q]               # Both fill? 
    # 
    # Wait, the model says "bid hit with prob..., ask likewise, independently".
    # If both hit, what happens?
    # "A fill earns its depth and moves inventory one unit".
    # Usually in these models, if both hit, it's a round trip: buy at bid, sell at ask.
    # Inventory: q -> q+1 (bid) -> q (ask). Net change 0.
    # Earnings: bid_depth + ask_depth.
    # Let's verify the "independently" part.
    # If both hit:
    #   Term: p_b * p_a * (b + a + V[n+1, q])
    # 
    # Let's re-read carefully: "bid hit with probability... ask likewise, independently."
    # "A fill earns its depth and moves inventory one unit".
    # So if bid hits: earn b, q -> q+1.
    # If ask hits: earn a, q -> q-1.
    # If both: earn b+a, q -> q+1-1 = q.
    
    # So the expected value update for a given (b, a) is:
    # E[V_next] = (1-p_b)(1-p_a)V[q] + p_b(1-p_a)V[q+1] + (1-p_b)p_a V[q-1] + p_b p_a V[q]
    #           = V[q] + p_b(1-p_a)(V[q+1]-V[q]) + (1-p_b)p_a(V[q-1]-V[q])
    #           = V[q] + p_b(V[q+1]-V[q]) - p_b p_a (V[q+1]-V[q]) + p_a(V[q-1]-V[q]) - p_b p_a (V[q-1]-V[q])
    #           = V[q] + p_b(1-p_a)(V[q+1]-V[q]) + p_a(1-p_b)(V[q-1]-V[q])
    
    # Plus immediate rewards:
    # R = b * p_b + a * p_a  (if fills happen)
    # Note: If both happen, we earn b AND a.
    # So R = b*p_b + a*p_a.
    
    # Plus inventory cost:
    # C = gamma/2 * sigma^2 * q^2 * dt
    
    # Total Cost to minimize:
    # J(b,a) = C + R + E[V_next]
    # Note: The objective is "depths earned minus sum...". So we want to MAXIMIZE Earnings - Cost.
    # Or MINIMIZE Cost - Earnings.
    # Let's define Value V as the maximum expected *net benefit* from this state onwards.
    # V[n, q] = max_{b,a} ( Earnings - InventoryCost + Expected Future Value )
    
    # Let's stick to Maximization.
    # V[n, q] = max_{b,a} [ (b*p_b + a*p_a) - (gamma/2 * sigma^2 * q^2 * dt) + E[V[n+1, q']] ]
    
    # Where E[V[n+1, q']] is the expected value of the next state.
    # q' is q+1 with prob p_b(1-p_a), q-1 with prob p_a(1-p_b), q with prob 1 - p_b(1-p_a) - p_a(1-p_b) = (1-p_b)(1-p_a) + p_b p_a.
    
    # Let's simplify E[V_next]:
    # P(q+1) = p_b(1-p_a)
    # P(q-1) = p_a(1-p_b)
    # P(q)   = 1 - P(q+1) - P(q-1)
    
    # V[n, q] = max_{b,a} [ b*p_b + a*p_a - C_inv + P(q+1)*V[n+1, q+1] + P(q-1)*V[n+1, q-1] + P(q)*V[n+1, q] ]
    
    # Boundary conditions:
    # If q == Q, p_b is effectively 0 for state transition (can't go to Q+1). 
    # But the problem says "no bid fill at q=Q". So if q=Q, p_b=0.
    # Similarly if q=-Q, p_a=0.
    
    # We iterate n from steps-1 down to 0.
    
    start_time = time.time()
    timeout = 250.0 # Safety margin for 300s limit
    
    for n in range(steps - 1, -1, -1):
        if time.time() - start_time > timeout:
             # If we run out of time, we return what we have (or a default)
             # But we need to finish the function. 
             # For now, break and return partial? No, we need full arrays.
             # We'll just let it run and hope it's fast enough. 
             # 1000 steps * 61 states * 50*50 depth combos = 152,500,000 operations.
             # NumPy vectorization can handle this if we vectorize over depths.
             
             pass
        
        # We vectorize the depth selection for all q at once? 
        # Or loop over q and vectorize over depths?
        # Looping over q (61 iterations) is fine.
        # Vectorizing over depths (2500 combos) is good.
        
        for q_idx in range(2 * Q + 1):
            q = q_idx - Q
            
            # Inventory cost
            C_inv = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Get next state values
            V_next_q = V[n+1, q_idx]
            
            # Handle boundaries for next state indices
            if q < Q:
                V_next_q_plus = V[n+1, q_idx + 1]
            else:
                V_next_q_plus = 0.0 # Unreachable or doesn't matter if prob is 0
                
            if q > -Q:
                V_next_q_minus = V[n+1, q_idx - 1]
            else:
                V_next_q_minus = 0.0
            
            # Compute probabilities for all depth pairs
            # p_b for each bid depth
            p_b = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)
            # p_a for each ask depth
            p_a = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)
            
            # Create meshgrid for all (b, a) combinations
            # p_b shape (n_depths,), p_a shape (n_depths,)
            # We want matrix P_b (n_depths, n_depths) where P_b[i,j] = p_b[i]
            # and P_a[i,j] = p_a[j]
            
            P_b = np.tile(p_b[:, np.newaxis], (1, n_depths))
            P_a = np.tile(p_a[np.newaxis, :], (n_depths, 1))
            
            # Probabilities of next states
            P_plus = P_b * (1.0 - P_a)
            P_minus = P_a * (1.0 - P_b)
            P_same = 1.0 - P_plus - P_minus
            
            # Expected Future Value
            # If q == Q, P_plus should be 0 (no bid fill). 
            # If q == -Q, P_minus should be 0 (no ask fill).
            if q == Q:
                P_plus = np.zeros_like(P_plus)
                P_same = P_same + P_plus # Mass moves to same? 
                # Actually if no bid fill, bid prob is 0.
                # So P_plus=0. P_same = (1-0)(1-p_a) + 0*p_a = 1-p_a.
                # P_minus = 0*(1-p_a) + (1-0)p_a = p_a.
                # Wait, my formulas:
                # P_plus = p_b(1-p_a). If p_b=0, P_plus=0.
                # P_minus = p_a(1-p_b). If p_b=0, P_minus=p_a.
                # P_same = 1 - 0 - p_a = 1-p_a.
                # This is consistent.
                
            if q == -Q:
                P_minus = np.zeros_like(P_minus)
                # p_a=0.
                # P_plus = p_b.
                # P_minus = 0.
                # P_same = 1-p_b.
                
            E_V_next = P_plus * V_next_q_plus + P_minus * V_next_q_minus + P_same * V_next_q
            
            # Immediate Reward
            # Earnings = b * p_b + a * p_a
            # depths_b shape (n_depths,), depths_a shape (n_depths,)
            Depths_b = np.tile(depths[:, np.newaxis], (1, n_depths))
            Depths_a = np.tile(depths[np.newaxis, :], (n_depths, 1))
            
            Rewards = Depths_b * P_b + Depths_a * P_a
            
            # Total Value for this (b,a) pair
            Values = Rewards - C_inv + E_V_next
            
            # Find max
            idx_max = np.unravel_index(np.argmax(Values), Values.shape)
            V[n, q_idx] = Values[idx_max]
            opt_bid[n, q_idx] = depths[idx_max[0]]
            opt_ask[n, q_idx] = depths[idx_max[1]]
            
    return opt_bid, opt_ask


def _get_policy(params):
    """
    Retrieves or computes the optimal policy for the given parameters.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    
    # Depth grid
    # We need to cover the range where optimal depth might lie.
    # 1/k is the neutral point.
    # Let's try a grid from 0.01 to 2.0/k?
    # Actually, let's just use a fixed reasonable grid.
    # If k=1.5, 1/k=0.66.
    # If k=0.75, 1/k=1.33.
    # Let's use a grid relative to 1/k.
    # Or just a fixed absolute grid since k is not too varied.
    # Range [0.01, 3.0] with 40 points.
    depth_grid = np.linspace(0.01, 3.0, 40)
    
    opt_bid, opt_ask = _compute_policy(
        params['sigma'], params['A'], params['k'], params['gamma'], 
        params['T'], params['steps'], params['Q'], depth_grid
    )
    
    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths based on pre-computed DP policy.
    """
    # Get the policy
    opt_bid, opt_ask = _get_policy(params)
    
    # t is current time. n is step index.
    # t = n * dt => n = t / dt
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    
    # Clamp n to valid range
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    # inventory is an array of ints
    # We need to map inventory q to index q_idx = q + Q
    Q = params['Q']
    q_idx = inventory + Q
    
    # Ensure indices are valid (though inventory should be in [-Q, Q])
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    # Retrieve depths
    # opt_bid[n, q_idx]
    bid_depths = opt_bid[n, q_idx]
    ask_depths = opt_ask[n, q_idx]
    
    # Return as float arrays
    return bid_depths.astype(float), ask_depths.astype(float)

# EVOLVE-BLOCK-END
