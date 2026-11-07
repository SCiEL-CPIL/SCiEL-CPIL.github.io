"""Optimal market making with inventory risk via dynamic programming."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for DP tables keyed by (sigma, A, k, gamma, T, steps, Q)
_dp_cache = {}


def _compute_dp_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the optimal bid and ask depths for each step and inventory level.
    
    Returns:
        bid_depths: array of shape (steps, 2*Q+1) - optimal bid depth at each step for each inventory
        ask_depths: array of shape (steps, 2*Q+1) - optimal ask depth at each step for each inventory
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    
    # We'll store the optimal depths for each step and inventory level
    # bid_depths[n, q_idx] = optimal bid depth at step n when inventory is q_idx - Q
    # ask_depths[n, q_idx] = optimal ask depth at step n when inventory is q_idx - Q
    
    bid_depths = np.zeros((steps, n_inv))
    ask_depths = np.zeros((steps, n_inv))
    
    # Value function: V[n, q] = expected optimal reward from step n to end given inventory q
    # q ranges from -Q to Q, index q_idx = q + Q
    # At the last step, no further rewards (terminal value = 0)
    
    # We work backwards from step steps-1 to step 0
    # V[steps, q] = 0 for all q (terminal)
    
    # For each step n (going backwards), for each inventory q, we choose bid_depth b and ask_depth a
    # to maximize:
    #   (1 - exp(-A * exp(-k*b) * dt)) * (-b) +   # bid fill: lose b (depth below mid)
    #   (1 - exp(-A * exp(-k*a) * dt)) * (a) +     # ask fill: gain a (depth above mid)
    #   (1 - p_bid) * (1 - p_ask) * V[n+1, q] +
    #   p_bid * (1 - p_ask) * V[n+1, q-1] +
    #   (1 - p_bid) * p_ask * V[n+1, q+1] +
    #   p_bid * p_ask * V[n+1, q] +
    #   -(gamma / 2) * sigma^2 * q^2 * dt
    #
    # where p_bid = 1 - exp(-A * exp(-k*b) * dt) if q < Q, else 0
    #       p_ask = 1 - exp(-A * exp(-k*a) * dt) if q > -Q, else 0
    #
    # Note: inventory q is in [-Q, Q]. If bid fill, inventory decreases by 1 (q -> q-1).
    #       If ask fill, inventory increases by 1 (q -> q+1).
    #       If both fill, inventory stays q (q-1+1 = q).
    #
    # Constraints: no bid fill at q = Q (so p_bid = 0 when q = Q)
    #              no ask fill at q = -Q (so p_ask = 0 when q = -Q)
    
    V_next = np.zeros(n_inv)  # V at step n+1
    
    # We need to search over possible depths. Let's use a grid of depths.
    # Depths should be positive (or zero). Let's use a reasonable range.
    # The fill probability is 1 - exp(-A * exp(-k*d) * dt)
    # For small d, exp(-k*d) ≈ 1, so p ≈ 1 - exp(-A*dt)
    # For large d, exp(-k*d) is small, so p is small.
    # Let's use depths from 0 to some max, say 10/k or so.
    
    # Actually, let's think about the scale. With k=1.5, 1/k ≈ 0.67.
    # Let's use a grid of depths.
    
    # For efficiency, let's use a moderate grid.
    # The optimal depth is likely not too large.
    
    # Let's use 200 depth levels for the search.
    n_depths = 200
    # Depth range: from 0 to some max
    # A reasonable max: when exp(-k*d) * A * dt is very small, the fill prob is negligible.
    # Let's say max_depth = 5/k to be safe.
    max_depth = 5.0 / k
    depth_grid = np.linspace(0, max_depth, n_depths)
    
    # Precompute fill probabilities for each depth
    # p_fill[d] = 1 - exp(-A * exp(-k*d) * dt)
    exp_kd = np.exp(-k * depth_grid)
    p_fill = 1.0 - np.exp(-A * exp_kd * dt)
    
    # Now do the DP backwards
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_inv)
        
        for q_idx in range(n_inv):
            q = q_idx - Q
            
            # Determine if bid/ask fills are allowed
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            # Inventory cost for this step
            inv_cost = -(gamma / 2.0) * sigma * sigma * q * q * dt
            
            # We need to maximize over b and a
            # This is a 2D optimization. Let's do it efficiently.
            
            # For each possible bid depth b (index i), compute the contribution
            # For each possible ask depth a (index j), compute the contribution
            
            # Let's precompute for each bid depth:
            #   p_b[i] = fill prob for bid
            #   reward_b[i] = -b (depth earned)
            
            # And for each ask depth:
            #   p_a[j] = fill prob for ask
            #   reward_a[j] = a (depth earned)
            
            # The total value for (i, j) is:
            #   p_b[i] * (-depth_grid[i]) + p_a[j] * depth_grid[j] +
            #   (1-p_b[i]) * (1-p_a[j]) * V_next[q_idx] +
            #   p_b[i] * (1-p_a[j]) * V_next[q_idx-1] +  # if q_idx > 0
            #   (1-p_b[i]) * p_a[j] * V_next[q_idx+1] +  # if q_idx < n_inv-1
            #   p_b[i] * p_a[j] * V_next[q_idx] +
            #   inv_cost
            #
            # Simplify:
            #   = p_b[i] * (-depth_grid[i]) + p_a[j] * depth_grid[j] +
            #   V_next[q_idx] * (1-p_b[i]) * (1-p_a[j]) +
            #   V_next[q_idx-1] * p_b[i] * (1-p_a[j]) +  # if valid
            #   V_next[q_idx+1] * (1-p_b[i]) * p_a[j] +  # if valid
            #   V_next[q_idx] * p_b[i] * p_a[j] +
            #   inv_cost
            #
            # = p_b[i] * (-depth_grid[i]) + p_a[j] * depth_grid[j] +
            #   V_next[q_idx] * [(1-p_b[i])*(1-p_a[j]) + p_b[i]*p_a[j]] +
            #   V_next[q_idx-1] * p_b[i] * (1-p_a[j]) +
            #   V_next[q_idx+1] * (1-p_b[i]) * p_a[j] +
            #   inv_cost
            
            best_val = -np.inf
            best_i = 0
            best_j = 0
            
            # Handle boundary cases
            V_left = V_next[q_idx - 1] if q_idx > 0 else -np.inf
            V_right = V_next[q_idx + 1] if q_idx < n_inv - 1 else -np.inf
            V_center = V_next[q_idx]
            
            # If can_bid is False, p_b = 0 always, so we only vary a
            # If can_ask is False, p_a = 0 always, so we only vary b
            
            if not can_bid and not can_ask:
                # No fills possible
                val = V_center + inv_cost
                V_curr[q_idx] = val
                bid_depths[n, q_idx] = depth_grid[0]  # 0 depth
                ask_depths[n, q_idx] = depth_grid[0]
                continue
            
            if not can_bid:
                # Only ask fills possible
                # Maximize over j (ask depth)
                # val[j] = p_a[j] * depth_grid[j] + V_center * (1-p_a[j]) + V_right * p_a[j] + inv_cost
                #        = p_a[j] * (depth_grid[j] + V_right - V_center) + V_center + inv_cost
                #        = p_a[j] * (depth_grid[j] + V_right - V_center) + V_center + inv_cost
                # Since V_right is -inf if q_idx == n_inv-1, but can_ask means q > -Q so q_idx > 0
                # Actually if not can_bid, q = Q, so q_idx = 2Q = n_inv-1
                # So V_right = -inf, meaning ask fill would go out of bounds? No, ask fill at q=Q...
                # Wait, can_ask = q > -Q. If q = Q, can_ask is True.
                # But if q = Q and ask fill, inventory becomes Q+1 which is out of bounds.
                # Hmm, the problem says "no ask fill at q = -Q". It doesn't say no ask fill at q = Q.
                # But inventory is bounded by Q. Let me re-read.
                # "no bid fill at q = Q, no ask fill at q = -Q"
                # So at q = Q, no bid fill (can't go above Q). At q = -Q, no ask fill (can't go below -Q).
                # But what about ask fill at q = Q? That would make it Q+1, which is out of bounds.
                # I think the constraint is that inventory stays in [-Q, Q]. So:
                # - At q = Q: no bid fill (would go to Q-1... wait, bid fill decreases inventory)
                #   Actually, bid fill: you sell, so inventory decreases. q -> q-1.
                #   Ask fill: you buy, so inventory increases. q -> q+1.
                #   At q = Q (max inventory), you can't buy more, so no ask fill.
                #   At q = -Q (min inventory), you can't sell more, so no bid fill.
                # 
                # Let me re-check: "A fill earns its depth and moves inventory one unit"
                # Bid hit: you sell at bid, so your inventory decreases by 1.
                # Ask hit: you buy at ask, so your inventory increases by 1.
                # "no bid fill at q = Q" - at max inventory, you can't sell? That doesn't make sense.
                # Unless q is defined differently. Let me re-read.
                # "inventory (4000,) ints in [-Q, Q]"
                # "no bid fill at q = Q, no ask fill at q = -Q"
                # I think q = Q means you have maximum inventory (all long), so you can't buy more (ask fill).
                # But the text says "no bid fill at q = Q". Maybe bid fill means you buy?
                # In market making, bid is the price you're willing to buy at. So a bid fill means
                # someone sells to you, increasing your inventory. Ask fill means someone buys from you,
                # decreasing your inventory.
                # So: bid fill -> q increases by 1. Ask fill -> q decreases by 1.
                # At q = Q (max), no bid fill (can't increase more).
                # At q = -Q (min), no ask fill (can't decrease more).
                # This makes sense!
                
                # So:
                # Bid fill: q -> q+1, allowed if q < Q
                # Ask fill: q -> q-1, allowed if q > -Q
                
                # Let me redo the DP with this understanding.
                
                # For now, let's just handle the simple case and I'll fix the logic.
                # Since this is getting complex, let me restart the DP logic.
                
                # Actually, let me just compute it numerically with the correct transitions.
                pass
            
            # Let me redo this more carefully.
            
            # Transitions:
            # Bid fill: q -> q+1, allowed if q < Q
            # Ask fill: q -> q-1, allowed if q > -Q
            
            # So:
            # can_bid = q < Q  (can have bid fill, inventory increases)
            # can_ask = q > -Q (can have ask fill, inventory decreases)
            
            # Wait, I had it backwards before. Let me be very careful.
            # In market making:
            # - You quote a bid (lower price) and an ask (higher price).
            # - If someone hits your bid, they sell to you. Your inventory increases by 1.
            # - If someone hits your ask, they buy from you. Your inventory decreases by 1.
            # 
            # So:
            # Bid fill: inventory increases. Allowed if q < Q (not at max).
            # Ask fill: inventory decreases. Allowed if q > -Q (not at min).
            
            # The problem statement says:
            # "no bid fill at q = Q, no ask fill at q = -Q"
            # This confirms: at q = Q, no bid fill (can't increase). At q = -Q, no ask fill (can't decrease).
            
            # So my original can_bid/can_ask were wrong. Let me fix:
            can_bid = (q < Q)   # bid fill increases inventory, allowed if not at max
            can_ask = (q > -Q)  # ask fill decreases inventory, allowed if not at min
            
            # V_left corresponds to q-1 (ask fill result), V_right corresponds to q+1 (bid fill result)
            # V_next[q_idx - 1] is V for inventory q-1 (after ask fill)
            # V_next[q_idx + 1] is V for inventory q+1 (after bid fill)
            
            V_ask_result = V_next[q_idx - 1] if q_idx > 0 else -np.inf  # q-1
            V_bid_result = V_next[q_idx + 1] if q_idx < n_inv - 1 else -np.inf  # q+1
            V_no_change = V_next[q_idx]  # q
            
            # Now maximize over b (bid depth) and a (ask depth)
            # Value = p_b * (-b) + p_a * a +
            #         (1-p_b)(1-p_a) * V_no_change +
            #         p_b * (1-p_a) * V_bid_result +   # bid fill only
            #         (1-p_b) * p_a * V_ask_result +   # ask fill only
            #         p_b * p_a * V_no_change +         # both fill
            #         inv_cost
            #
            # = p_b * (-b) + p_a * a +
            #   V_no_change * [(1-p_b)(1-p_a) + p_b*p_a] +
            #   V_bid_result * p_b * (1-p_a) +
            #   V_ask_result * (1-p_b) * p_a +
            #   inv_cost
            
            # For efficiency, let's iterate over b and for each b, find best a.
            # Or use vectorization.
            
            # Let's use a 2D approach with vectorization over depths.
            # This might be slow, but let's try.
            
            # Actually, for 1000 steps * 61 inventory levels * 200*200 depth combos = 242 million operations.
            # That's too slow. We need to be smarter.
            
            # Let's use a smaller depth grid, say 50 levels.
            # 1000 * 61 * 50 * 50 = 15.25 million. Still might be slow in Python.
            
            # Let's use 30 depth levels.
            # 1000 * 61 * 30 * 30 = 5.5 million. Better.
            
            # Or we can vectorize over the depth grid.
            
            # Let's try a different approach: for each (n, q), vectorize over all (i, j) pairs.
            
            # Actually, let's just use a coarse grid and hope it's fast enough.
            # Let's use 20 depth levels.
            
            n_depths_coarse = 20
            depth_grid_coarse = np.linspace(0, max_depth, n_depths_coarse)
            exp_kd_coarse = np.exp(-k * depth_grid_coarse)
            p_fill_coarse = 1.0 - np.exp(-A * exp_kd_coarse * dt)
            
            # For each possible bid depth i and ask depth j, compute the value
            # We'll use broadcasting
            pb = p_fill_coarse if can_bid else np.zeros(n_depths_coarse)
            pa = p_fill_coarse if can_ask else np.zeros(n_depths_coarse)
            db = depth_grid_coarse
            da = depth_grid_coarse
            
            # Create 2D grids
            pb_2d = np.tile(pb[:, np.newaxis], (1, n_depths_coarse))  # shape (n_depths, n_depths)
            pa_2d = np.tile(pa[np.newaxis, :], (n_depths_coarse, 1))  # shape (n_depths, n_depths)
            db_2d = np.tile(db[:, np.newaxis], (1, n_depths_coarse))
            da_2d = np.tile(da[np.newaxis, :], (n_depths_coarse, 1))
            
            # Compute value matrix
            # Value[i,j] = pb[i]*(-db[i]) + pa[j]*da[j] +
            #              V_no_change * ((1-pb[i])(1-pa[j]) + pb[i]*pa[j]) +
            #              V_bid_result * pb[i] * (1-pa[j]) +
            #              V_ask_result * (1-pb[i]) * pa[j] +
            #              inv_cost
            
            val_matrix = (pb_2d * (-db_2d) + pa_2d * da_2d +
                          V_no_change * ((1 - pb_2d) * (1 - pa_2d) + pb_2d * pa_2d) +
                          V_bid_result * pb_2d * (1 - pa_2d) +
                          V_ask_result * (1 - pb_2d) * pa_2d +
                          inv_cost)
            
            # Handle -inf cases: if V_bid_result is -inf and pb_2d > 0, that entry should be -inf
            # Similarly for V_ask_result
            if np.isneginf(V_bid_result):
                val_matrix[pb_2d > 0] = -np.inf
            if np.isneginf(V_ask_result):
                val_matrix[pa_2d > 0] = -np.inf
            
            # Find the best (i, j)
            flat_idx = np.argmax(val_matrix)
            best_i, best_j = np.unravel_index(flat_idx, val_matrix.shape)
            
            V_curr[q_idx] = val_matrix[best_i, best_j]
            bid_depths[n, q_idx] = db[best_i]
            ask_depths[n, q_idx] = da[best_j]
        
        V_next = V_curr
    
    return bid_depths, ask_depths


def _get_dp_table(sigma, A, k, gamma, T, steps, Q):
    """Get or compute the DP table for the given parameters."""
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _dp_cache:
        _dp_cache[key] = _compute_dp_table(sigma, A, k, gamma, T, steps, Q)
    return _dp_cache[key]


def quote(t, inventory, mid, params):
    """
    Quote using precomputed DP table.
    
    Args:
        t: current time
        inventory: array of shape (4000,) with inventory levels
        mid: array of shape (4000,) with mid prices
        params: dict with 'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'
    
    Returns:
        (bid_depths, ask_depths): tuple of two float arrays of shape (4000,)
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Get the DP table
    bid_table, ask_table = _get_dp_table(sigma, A, k, gamma, T, steps, Q)
    
    # Determine current step
    dt = T / steps
    n = int(round(t / dt))
    n = np.clip(n, 0, steps - 1)
    
    # Map inventory to index: q_idx = q + Q
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    # Look up depths
    bid_depths = bid_table[n, q_idx]
    ask_depths = ask_table[n, q_idx]
    
    # Ensure no negative depths and valid values
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    # Check for invalid values
    bid_depths = np.where(np.isnan(bid_depths) | np.isneginf(bid_depths), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | np.isneginf(ask_depths), np.inf, ask_depths)
    
    # Values below -3/k are invalid; our depths should be >= 0, so this shouldn't happen
    # But let's be safe
    min_valid = -3.0 / k
    bid_depths = np.where(bid_depths < min_valid, np.inf, bid_depths)
    ask_depths = np.where(ask_depths < min_valid, np.inf, ask_depths)
    
    return bid_depths.astype(np.float64), ask_depths.astype(np.float64)
# EVOLVE-BLOCK-END
