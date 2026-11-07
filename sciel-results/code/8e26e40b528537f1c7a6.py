"""
Strategy: Pre-compute the exact optimal policy using Dynamic Programming (Bellman Equation).
The state space is small (1000 steps x 61 inventory levels), so we can solve for the
optimal bid/ask depths for each state. We cache the results per regime (sigma, A, k, gamma)
so that the first call for a regime pays the DP cost, and subsequent calls are O(1) lookups.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for pre-computed optimal policies
# Key: tuple of regime parameters
# Value: (bid_depths, ask_depths) arrays of shape (steps, 2*Q+1)
_policy_cache = {}

def _compute_optimal_policy(params):
    """
    Compute the optimal bid and ask depths for all (t, q) states using DP.
    Returns:
        bid_depths: array of shape (steps, 2*Q+1)
        ask_depths: array of shape (steps, 2*Q+1)
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    n_inventory_levels = 2 * Q + 1
    
    # bid_depths[step, inv_index] = optimal bid depth at step, inventory
    # ask_depths[step, inv_index] = optimal ask depth at step, inventory
    bid_depths = np.zeros((steps, n_inventory_levels))
    ask_depths = np.zeros((steps, n_inventory_levels))
    
    # Value function: V[step, inv_index] = optimal expected reward from step to end
    # V[steps, inv_index] = 0 (no terminal charge)
    V = np.zeros((steps + 1, n_inventory_levels))
    
    # We work backwards from step = steps-1 to step = 0
    # At each step, we choose bid_depth and ask_depth to maximize expected reward + future value
    
    # For numerical stability and to avoid infinite search, we limit the search range for depths
    # The depth should be positive and reasonable. Let's search in [0.001, 10.0] with fine steps.
    # Actually, the optimal depth is likely around 1/k. Let's be smarter.
    
    # The probability of a bid hit is 1 - exp(-A * exp(-k * bid_depth) * dt)
    # If bid_depth is very large, exp(-k * bid_depth) is near 0, so prob is near 0.
    # If bid_depth is very small (near 0), exp(-k * bid_depth) is near 1, so prob is 1 - exp(-A * dt).
    # The reward from a bid hit is bid_depth (we earn the depth).
    # So we want to balance: higher depth = more reward per fill, but lower probability of fill.
    
    # Let's do a grid search over possible depths.
    # The relevant range for depth: if k=1.5, 1/k ≈ 0.67. Let's search [0.01, 5.0].
    # But we need to be careful: the optimal depth depends on inventory (risk aversion).
    
    # For efficiency, let's use a coarse grid first, then refine if needed.
    # Actually, with 1000 steps and 61 inventory levels, and if we search over ~100 depth values,
    # that's 1000 * 61 * 100 = 6.1 million evaluations, which is feasible.
    
    # Let's define a grid of candidate depths.
    # The depth affects: reward (linear in depth) and probability (exponential in -k*depth).
    # Let's use a logarithmic or linear grid.
    
    # Start with a linear grid from 0.01 to 5.0 with 100 points.
    depth_grid = np.linspace(0.01, 5.0, 100)
    
    # Precompute exp(-k * depth_grid) for efficiency
    exp_factor = np.exp(-k * depth_grid)  # shape: (100,)
    prob_factor = 1 - np.exp(-A * exp_factor * dt)  # shape: (100,)
    
    for step in range(steps - 1, -1, -1):
        for inv in range(-Q, Q + 1):
            inv_index = inv + Q
            
            best_val = -np.inf
            best_bid = 0.01
            best_ask = 0.01
            
            # If inv == Q, no bid fill possible, so bid_depth doesn't matter for reward,
            # but it still affects the state transition (no transition from bid).
            # Actually, if inv == Q, a bid fill would make inv = Q+1 which is out of bounds.
            # The problem says "no bid fill at q = Q", so if inv == Q, bid hits are ignored.
            # Similarly, no ask fill at q = -Q.
            
            can_bid = (inv < Q)
            can_ask = (inv > -Q)
            
            # We need to choose bid_depth and ask_depth to maximize:
            # E[reward] + E[V[step+1, new_inv]]
            
            # The reward from this step:
            # - (gamma/2) * sigma^2 * inv^2 * dt  (inventory cost, always incurred)
            # + bid_depth * I(bid hit)  (if bid hits, we earn bid_depth)
            # + ask_depth * I(ask hit)  (if ask hits, we earn ask_depth)
            
            # Wait, the problem says "depths earned". So if a bid is hit, we earn the bid_depth.
            # If an ask is hit, we earn the ask_depth.
            
            # The inventory cost is (gamma/2) * sigma^2 * inv^2 * dt, subtracted.
            
            inv_cost = (gamma / 2.0) * sigma**2 * inv**2 * dt
            
            # We iterate over candidate bid and ask depths.
            # For efficiency, we can precompute the expected future value for each (bid_depth, ask_depth) pair.
            
            # Let's vectorize over the depth grid.
            # For each candidate bid_depth (index i) and ask_depth (index j):
            #   p_bid = prob_factor[i] if can_bid else 0
            #   p_ask = prob_factor[j] if can_ask else 0
            #   
            #   Possible outcomes:
            #   1. No hit: prob = (1-p_bid)*(1-p_ask), new_inv = inv
            #   2. Bid hit only: prob = p_bid*(1-p_ask), new_inv = inv - 1 (if can_bid)
            #   3. Ask hit only: prob = (1-p_bid)*p_ask, new_inv = inv + 1 (if can_ask)
            #   4. Both hit: prob = p_bid*p_ask, new_inv = inv (bid and ask cancel? or net 0?)
            
            # The problem says "A fill earns its depth and moves inventory one unit".
            # It doesn't explicitly say what happens if both hit. Let's assume they are independent
            # and if both hit, inventory doesn't change (one buy and one sell cancel out).
            
            # Actually, re-reading: "bid hit with probability ..., ask likewise, independently."
            # So if both hit, we have a buy and a sell, net inventory change is 0.
            
            # Let's compute the expected value for each (i, j) pair.
            
            # For efficiency, let's precompute the future values for inv-1, inv, inv+1.
            V_next_inv = V[step + 1, inv_index]
            V_next_inv_minus = V[step + 1, inv_index - 1] if (inv > -Q) else 0.0
            V_next_inv_plus = V[step + 1, inv_index + 1] if (inv < Q) else 0.0
            
            # Now, for each (i, j):
            #   p_b = prob_factor[i] if can_bid else 0
            #   p_a = prob_factor[j] if can_ask else 0
            #   
            #   E[V_next] = (1-p_b)*(1-p_a)*V_next_inv + p_b*(1-p_a)*V_next_inv_minus + (1-p_b)*p_a*V_next_inv_plus + p_b*p_a*V_next_inv
            #              = V_next_inv * (1 - p_b*p_a - p_b*(1-p_a) - (1-p_b)*p_a + p_b*p_a) ... let me just compute directly.
            #              = (1-p_b)*(1-p_a)*V_next_inv + p_b*(1-p_a)*V_next_inv_minus + (1-p_b)*p_a*V_next_inv_plus + p_b*p_a*V_next_inv
            
            #   Reward = inv_cost + p_b * depth_grid[i] + p_a * depth_grid[j]
            
            #   Total = Reward + E[V_next]
            
            # Let's vectorize this.
            
            if can_bid and can_ask:
                # p_b and p_a vary with i and j
                p_b = prob_factor  # shape: (100,)
                p_a = prob_factor  # shape: (100,)
                
                # We need to compute for all (i, j) pairs.
                # This is a 2D grid of 100x100 = 10000 values.
                
                # Let's use broadcasting.
                p_b_col = p_b[:, np.newaxis]  # shape: (100, 1)
                p_a_row = p_a[np.newaxis, :]  # shape: (1, 100)
                
                # E[V_next]
                term1 = (1 - p_b_col) * (1 - p_a_row) * V_next_inv  # shape: (100, 100)
                term2 = p_b_col * (1 - p_a_row) * V_next_inv_minus  # shape: (100, 100)
                term3 = (1 - p_b_col) * p_a_row * V_next_inv_plus  # shape: (100, 100)
                term4 = p_b_col * p_a_row * V_next_inv  # shape: (100, 100)
                
                E_V_next = term1 + term2 + term3 + term4
                
                # Reward
                reward = inv_cost + p_b_col * depth_grid[:, np.newaxis] + p_a_row * depth_grid[np.newaxis, :]
                
                total = reward + E_V_next
                
                # Find the max
                idx = np.unravel_index(np.argmax(total), total.shape)
                best_val = total[idx]
                best_bid = depth_grid[idx[0]]
                best_ask = depth_grid[idx[1]]
            
            elif can_bid:
                # No ask fill possible
                p_b = prob_factor  # shape: (100,)
                p_a = np.zeros(1)  # no ask
                
                # For each bid depth i, ask depth doesn't matter (no ask fill).
                # But we still need to choose an ask_depth. Since no ask fill, the ask_depth
                # doesn't affect the reward or the state transition. So we can choose any ask_depth.
                # Let's choose a small value to be safe.
                
                # E[V_next] = (1-p_b)*V_next_inv + p_b*V_next_inv_minus
                E_V_next = (1 - p_b) * V_next_inv + p_b * V_next_inv_minus  # shape: (100,)
                
                # Reward = inv_cost + p_b * depth_grid[i]
                reward = inv_cost + p_b * depth_grid  # shape: (100,)
                
                total = reward + E_V_next
                
                idx = np.argmax(total)
                best_val = total[idx]
                best_bid = depth_grid[idx]
                best_ask = 0.01  # arbitrary, since no ask fill
            
            elif can_ask:
                # No bid fill possible
                p_a = prob_factor  # shape: (100,)
                p_b = np.zeros(1)  # no bid
                
                # E[V_next] = (1-p_a)*V_next_inv + p_a*V_next_inv_plus
                E_V_next = (1 - p_a) * V_next_inv + p_a * V_next_inv_plus  # shape: (100,)
                
                # Reward = inv_cost + p_a * depth_grid[j]
                reward = inv_cost + p_a * depth_grid  # shape: (100,)
                
                total = reward + E_V_next
                
                idx = np.argmax(total)
                best_val = total[idx]
                best_bid = 0.01  # arbitrary, since no bid fill
                best_ask = depth_grid[idx]
            
            else:
                # Neither bid nor ask fill possible (shouldn't happen since -Q < Q)
                best_val = inv_cost + V_next_inv
                best_bid = 0.01
                best_ask = 0.01
            
            bid_depths[step, inv_index] = best_bid
            ask_depths[step, inv_index] = best_ask
            V[step, inv_index] = best_val
    
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """
    Return the optimal bid and ask depths for each path.
    """
    # Get or compute the optimal policy
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        # Check time budget
        start_time = time.time()
        bid_depths, ask_depths = _compute_optimal_policy(params)
        _policy_cache[key] = (bid_depths, ask_depths)
    
    bid_depths, ask_depths = _policy_cache[key]
    
    # Get the current step
    dt = params['T'] / params['steps']
    step = int(round(t / dt))
    step = min(step, params['steps'] - 1)
    step = max(step, 0)
    
    # Get the inventory indices
    Q = params['Q']
    inv_indices = inventory + Q
    
    # Look up the optimal depths
    bid_result = bid_depths[step, inv_indices]
    ask_result = ask_depths[step, inv_indices]
    
    return bid_result, ask_result

# EVOLVE-BLOCK-END
