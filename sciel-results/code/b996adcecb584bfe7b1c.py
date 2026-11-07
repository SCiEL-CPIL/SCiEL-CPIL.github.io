"""Optimal market making with inventory risk via discrete DP policy precomputation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# --- Precomputation of optimal policies for each regime ---

def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the optimal bid/ask depth policy via backward induction on a discrete state grid.
    
    Returns:
        bid_table: array of shape (steps, 2*Q+1) with optimal bid depth for each (time, inventory)
        ask_table: array of shape (steps, 2*Q+1) with optimal ask depth for each (time, inventory)
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    inv_grid = np.arange(-Q, Q + 1)  # -Q, -Q+1, ..., Q
    
    # Depth grid: we search over possible depths
    # The depth must be in (0, 3/k) to be valid
    max_depth = 3.0 / k
    # Use a fine grid for depth search
    n_depth = 200
    depth_grid = np.linspace(0.01, max_depth - 0.01, n_depth)
    
    # Precompute fill probabilities for each depth
    # p_fill(d) = 1 - exp(-A * exp(-k * d) * dt)
    p_fill_grid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # Value function: V[n, i] where i indexes inventory from -Q to Q
    # We'll store V for current and previous step
    V_next = np.zeros(n_inv)  # Terminal: no charge
    
    bid_table = np.zeros((steps, n_inv))
    ask_table = np.zeros((steps, n_inv))
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_inv)
        
        for i in range(n_inv):
            q = inv_grid[i]
            
            # Inventory cost for this step
            inv_cost = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            best_bid_val = -np.inf
            best_bid_depth = 0.0
            best_ask_val = -np.inf
            best_ask_depth = 0.0
            
            # If q == Q, no bid fill possible
            if q < Q:
                for di in range(n_depth):
                    d_bid = depth_grid[di]
                    p_bid = p_fill_grid[di]
                    
                    # If bid fills: inventory becomes q+1, earn d_bid
                    new_q_bid = q + 1
                    if new_q_bid <= Q:
                        idx_next = new_q_bid + Q  # index in inv_grid
                        if idx_next >= 0 and idx_next < n_inv:
                            val_bid = d_bid + (1 - p_bid) * V_next[i] + p_bid * V_next[idx_next]
                        else:
                            val_bid = d_bid + (1 - p_bid) * V_next[i]
                    else:
                        # Should not happen since we check q < Q
                        val_bid = (1 - p_bid) * V_next[i]
                    
                    if val_bid > best_bid_val:
                        best_bid_val = val_bid
                        best_bid_depth = d_bid
            
            # If q == -Q, no ask fill possible
            if q > -Q:
                for di in range(n_depth):
                    d_ask = depth_grid[di]
                    p_ask = p_fill_grid[di]
                    
                    # If ask fills: inventory becomes q-1, earn d_ask
                    new_q_ask = q - 1
                    if new_q_ask >= -Q:
                        idx_next = new_q_ask + Q
                        if idx_next >= 0 and idx_next < n_inv:
                            val_ask = d_ask + (1 - p_ask) * V_next[i] + p_ask * V_next[idx_next]
                        else:
                            val_ask = d_ask + (1 - p_ask) * V_next[i]
                    else:
                        val_ask = (1 - p_ask) * V_next[i]
                    
                    if val_ask > best_ask_val:
                        best_ask_val = val_ask
                        best_ask_depth = d_ask
            
            # The total value is the sum of independent bid and ask decisions
            # Since bid and ask fills are independent:
            # E[reward] = E[bid_reward] + E[ask_reward]
            # But we need to be careful: if both fill, inventory changes by 2?
            # Actually, the problem says "bid hit with probability ..., ask likewise, independently"
            # So both can fill in the same step.
            
            # Let's redo this properly with joint consideration.
            # Actually, for the value function:
            # V = -inv_cost + E[depth_earned + V_next[new_inventory]]
            # where depth_earned = d_bid * I(bid_fills) + d_ask * I(ask_fills)
            # and new_inventory = q + I(bid_fills) - I(ask_fills)
            
            # So we need to consider all 4 cases:
            # (no bid, no ask): prob (1-p_b)(1-p_a), reward 0, new_q = q
            # (bid only): prob p_b(1-p_a), reward d_b, new_q = q+1
            # (ask only): prob (1-p_b)p_a, reward d_a, new_q = q-1
            # (both): prob p_b*p_a, reward d_b+d_a, new_q = q
            
            # But wait, we choose d_b and d_a simultaneously.
            # This is a 2D optimization. Let's do it properly.
            
            best_total_val = -np.inf
            best_bid_depth = 0.0
            best_ask_depth = 0.0
            
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            if can_bid and can_ask:
                # 2D search
                for di_b in range(n_depth):
                    d_b = depth_grid[di_b]
                    p_b = p_fill_grid[di_b]
                    for di_a in range(n_depth):
                        d_a = depth_grid[di_a]
                        p_a = p_fill_grid[di_a]
                        
                        # Case 1: no fills
                        val1 = V_next[i]
                        # Case 2: bid only
                        idx_b = (q + 1) + Q
                        val2 = d_b + V_next[idx_b] if 0 <= idx_b < n_inv else d_b
                        # Case 3: ask only
                        idx_a = (q - 1) + Q
                        val3 = d_a + V_next[idx_a] if 0 <= idx_a < n_inv else d_a
                        # Case 4: both
                        val4 = d_b + d_a + V_next[i]
                        
                        total = ((1 - p_b) * (1 - p_a) * val1 +
                                 p_b * (1 - p_a) * val2 +
                                 (1 - p_b) * p_a * val3 +
                                 p_b * p_a * val4)
                        
                        if total > best_total_val:
                            best_total_val = total
                            best_bid_depth = d_b
                            best_ask_depth = d_a
            elif can_bid:
                for di_b in range(n_depth):
                    d_b = depth_grid[di_b]
                    p_b = p_fill_grid[di_b]
                    idx_b = (q + 1) + Q
                    val_b = d_b + V_next[idx_b] if 0 <= idx_b < n_inv else d_b
                    total = (1 - p_b) * V_next[i] + p_b * val_b
                    if total > best_total_val:
                        best_total_val = total
                        best_bid_depth = d_b
                        best_ask_depth = 0.0
            elif can_ask:
                for di_a in range(n_depth):
                    d_a = depth_grid[di_a]
                    p_a = p_fill_grid[di_a]
                    idx_a = (q - 1) + Q
                    val_a = d_a + V_next[idx_a] if 0 <= idx_a < n_inv else d_a
                    total = (1 - p_a) * V_next[i] + p_a * val_a
                    if total > best_total_val:
                        best_total_val = total
                        best_bid_depth = 0.0
                        best_ask_depth = d_a
            else:
                best_total_val = V_next[i]
                best_bid_depth = 0.0
                best_ask_depth = 0.0
            
            V_curr[i] = best_total_val - inv_cost
            bid_table[n, i] = best_bid_depth
            ask_table[n, i] = best_ask_depth
        
        V_next = V_curr
    
    return bid_table, ask_table


# Cache for precomputed policies
_policy_cache = {}

def _get_policy(params):
    """Get or compute the policy for the given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _policy_cache:
        _policy_cache[key] = _compute_policy(
            params['sigma'], params['A'], params['k'], params['gamma'],
            params['T'], params['steps'], params['Q']
        )
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Quote optimal bid/ask depths based on precomputed DP policy.
    
    Args:
        t: current time
        inventory: array of shape (4000,) with integer inventories in [-Q, Q]
        mid: array of shape (4000,) with mid prices
        params: dict with keys 'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'
    
    Returns:
        (bid_depths, ask_depths): tuple of two arrays of shape (4000,)
    """
    Q = params['Q']
    steps = params['steps']
    
    # Determine current step
    dt = params['T'] / steps
    step = int(round(t / dt))
    step = np.clip(step, 0, steps - 1)
    
    # Get precomputed policy
    bid_table, ask_table = _get_policy(params)
    
    # Map inventory to table index: inventory in [-Q, Q] -> index in [0, 2Q]
    inv_idx = inventory + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Look up depths
    bid_depths = bid_table[step, inv_idx]
    ask_depths = ask_table[step, inv_idx]
    
    # Handle cases where depth is 0 (no quote) -> set to inf
    bid_depths = np.where(bid_depths > 0, bid_depths, np.inf)
    ask_depths = np.where(ask_depths > 0, ask_depths, np.inf)
    
    # Ensure validity: depths must be in (0, 3/k)
    max_depth = 3.0 / params['k']
    bid_depths = np.clip(bid_depths, 0.01, max_depth - 0.01)
    ask_depths = np.clip(ask_depths, 0.01, max_depth - 0.01)
    # Re-apply inf for no-quote
    bid_depths = np.where(bid_depths == 0, np.inf, bid_depths)
    ask_depths = np.where(ask_depths == 0, np.inf, ask_depths)
    
    # Ensure no NaN or -inf
    bid_depths = np.nan_to_num(bid_depths, nan=np.inf, posinf=np.inf, neginf=np.inf)
    ask_depths = np.nan_to_num(ask_depths, nan=np.inf, posinf=np.inf, neginf=np.inf)
    
    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
