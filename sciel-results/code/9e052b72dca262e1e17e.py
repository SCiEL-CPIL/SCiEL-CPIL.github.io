"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds of safety margin

# Precompute optimal base depth factor
# Maximize d * (1 - exp(-k*d)) -> k*d is root of 1 - exp(-y) = y*exp(-y)
# y* = 1.59362426...
_OPTIMAL_KD = 1.59362426
_BASE_EXP = np.exp(_OPTIMAL_KD)  # ~4.92

# Cache for precomputed optimal policies per (sigma, A, k, gamma, T, steps, Q)
_dp_cache = {}

def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depth arrays for each inventory level and time step
    using backward induction on a coarse grid.
    
    Returns:
        bid_depth_table: shape (steps+1, 2*Q+1) - optimal bid depth for each t, q
        ask_depth_table: shape (steps+1, 2*Q+1) - optimal ask depth for each t, q
    """
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key in _dp_cache:
        return _dp_cache[cache_key]
    
    dt = T / steps
    num_q = 2 * Q + 1
    q_indices = np.arange(-Q, Q + 1)  # -Q, ..., Q
    
    # Initialize value function V[n, q] where n is time step index, q is inventory index
    # V[n, q] = optimal expected value from time step n with inventory q
    
    # We'll use a grid of possible depths for each state
    # The optimal depth is the one that maximizes:
    # d * (1 - exp(-k*d)) + (1 - p_fill) * V[n+1, q+1] + p_fill * V[n+1, q]
    # for bid (increases inventory), and similarly for ask
    
    # Actually, let's think about this more carefully.
    # At each step, we choose bid_depth and ask_depth.
    # The probability of a bid fill is p_bid = 1 - exp(-A * exp(-k * bid_depth) * dt)
    # If a bid fill occurs, we earn bid_depth and inventory increases by 1.
    # The probability of an ask fill is p_ask = 1 - exp(-A * exp(-k * ask_depth) * dt)
    # If an ask fill occurs, we earn ask_depth and inventory decreases by 1.
    
    # The expected value at state (n, q) is:
    # V[n, q] = max_{bid_depth, ask_depth} [
    #   (1 - p_bid) * (1 - p_ask) * (V[n+1, q] - 0.5*gamma*sigma^2*q^2*dt)
    #   + p_bid * (1 - p_ask) * (bid_depth + V[n+1, q+1] - 0.5*gamma*sigma^2*(q+1)^2*dt)
    #   + (1 - p_bid) * p_ask * (ask_depth + V[n+1, q-1] - 0.5*gamma*sigma^2*(q-1)^2*dt)
    #   + p_bid * p_ask * (bid_depth + ask_depth + V[n+1, q] - 0.5*gamma*sigma^2*q^2*dt)
    # ]
    
    # This is complex. Let's use a simpler approach: assume bid and ask decisions are independent.
    # For bid: choose bid_depth to maximize
    #   p_bid * bid_depth + (1 - p_bid) * 0 + [expected continuation value]
    # The continuation value depends on whether a fill occurs.
    
    # Let's use value iteration with a discretized depth grid.
    # We'll precompute a grid of candidate depths.
    
    # Candidate depths: from 0 to some max, plus inf (no quote)
    # Let's use a logarithmic grid
    depth_grid = np.concatenate([
        [0.0],
        np.linspace(0.01, 5.0/k, 50),
        [np.inf]
    ])
    num_depths = len(depth_grid)
    
    # Initialize value function to zeros
    V = np.zeros((steps + 1, num_q), dtype=np.float64)
    
    # Store optimal depths
    bid_depth_table = np.zeros((steps + 1, num_q), dtype=np.float64)
    ask_depth_table = np.zeros((steps + 1, num_q), dtype=np.float64)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        for qi in range(num_q):
            q = q_indices[qi]
            
            # If at boundary, we can't quote that side
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            # Compute optimal bid depth
            if can_bid:
                best_bid_val = -np.inf
                best_bid_depth = 0.0
                
                for d in depth_grid:
                    if np.isinf(d):
                        p_fill = 0.0
                        reward = 0.0
                    else:
                        p_fill = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
                        reward = d * p_fill
                    
                    # Continuation value:
                    # If fill: q+1, else q
                    # But we also need to consider ask side...
                    # This is getting complex. Let's use a simpler heuristic.
                    
                    # For now, use a greedy approach:
                    # Maximize immediate reward + discounted future value
                    # This is approximate but should be good enough.
                    
                    # Expected future value if fill: V[n+1, q+1]
                    # Expected future value if no fill: V[n+1, q]
                    q_next_fill = min(q + 1, Q)
                    qi_next_fill = q_next_fill + Q
                    future_fill = V[n + 1, qi_next_fill]
                    future_no_fill = V[n + 1, qi]
                    
                    total_val = reward + p_fill * future_fill + (1 - p_fill) * future_no_fill
                    
                    if total_val > best_bid_val:
                        best_bid_val = total_val
                        best_bid_depth = d
                
                bid_depth_table[n, qi] = best_bid_depth
            else:
                bid_depth_table[n, qi] = np.inf
            
            # Compute optimal ask depth
            if can_ask:
                best_ask_val = -np.inf
                best_ask_depth = 0.0
                
                for d in depth_grid:
                    if np.isinf(d):
                        p_fill = 0.0
                        reward = 0.0
                    else:
                        p_fill = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
                        reward = d * p_fill
                    
                    q_next_fill = max(q - 1, -Q)
                    qi_next_fill = q_next_fill + Q
                    future_fill = V[n + 1, qi_next_fill]
                    future_no_fill = V[n + 1, qi]
                    
                    total_val = reward + p_fill * future_fill + (1 - p_fill) * future_no_fill
                    
                    if total_val > best_ask_val:
                        best_ask_val = total_val
                        best_ask_depth = d
                
                ask_depth_table[n, qi] = best_ask_depth
            else:
                ask_depth_table[n, qi] = np.inf
            
            # Update value function
            # Use the optimal depths we just found
            bid_d = bid_depth_table[n, qi]
            ask_d = ask_depth_table[n, qi]
            
            if np.isinf(bid_d):
                p_bid = 0.0
                bid_reward = 0.0
            else:
                p_bid = 1.0 - np.exp(-A * np.exp(-k * bid_d) * dt)
                bid_reward = bid_d * p_bid
            
            if np.isinf(ask_d):
                p_ask = 0.0
                ask_reward = 0.0
            else:
                p_ask = 1.0 - np.exp(-A * np.exp(-k * ask_d) * dt)
                ask_reward = ask_d * p_ask
            
            # Four cases: no fill, bid fill, ask fill, both fill
            # Inventory changes: 0, +1, -1, 0 respectively
            q0 = q
            q1 = min(q + 1, Q)
            q2 = max(q - 1, -Q)
            
            qi0 = q0 + Q
            qi1 = q1 + Q
            qi2 = q2 + Q
            
            # Inventory penalty
            inv_penalty = 0.5 * gamma * sigma * sigma * q * q * dt
            
            # Expected value
            V[n, qi] = (
                (1 - p_bid) * (1 - p_ask) * (V[n + 1, qi0] - inv_penalty) +
                p_bid * (1 - p_ask) * (bid_reward + V[n + 1, qi1] - 0.5 * gamma * sigma * sigma * q1 * q1 * dt) +
                (1 - p_bid) * p_ask * (ask_reward + V[n + 1, qi2] - 0.5 * gamma * sigma * sigma * q2 * q2 * dt) +
                p_bid * p_ask * (bid_reward + ask_reward + V[n + 1, qi0] - inv_penalty)
            )
    
    _dp_cache[cache_key] = (bid_depth_table, ask_depth_table)
    return bid_depth_table, ask_depth_table


def quote(t, inventory, mid, params):
    """
    Inventory-aware market making strategy using precomputed optimal policy.
    """
    N = len(inventory)
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Deadline guard: if we're close to the limit, return a simple safe quote
    elapsed = time.time() - _start_time
    if elapsed > 325.0:  # 330s limit minus 5s margin
        # Fallback to simple linear skew with optimal base
        base_depth = _OPTIMAL_KD / k
        inv_float = inventory.astype(np.float64)
        t_rem = T - t
        skew = gamma * sigma * sigma * t_rem * inv_float
        bid_depths = base_depth + skew
        ask_depths = base_depth - skew
        bid_depths = np.clip(bid_depths, -3.0/k, 100.0/k)
        ask_depths = np.clip(ask_depths, -3.0/k, 100.0/k)
        bid_depths[inventory == Q] = np.inf
        ask_depths[inventory == -Q] = np.inf
        return bid_depths, ask_depths
    
    # Get or compute the optimal policy table
    bid_depth_table, ask_depth_table = _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q)
    
    # Determine current time step index
    # t = n * dt, so n = t / dt
    n = int(round(t / dt))
    n = min(max(n, 0), steps)
    
    # Get depths for each inventory level
    # inventory is in [-Q, Q], so index is inventory + Q
    inv_indices = inventory + Q
    
    # Clip indices to valid range
    inv_indices = np.clip(inv_indices, 0, 2 * Q)
    
    bid_depths = bid_depth_table[n, inv_indices].copy()
    ask_depths = ask_depth_table[n, inv_indices].copy()
    
    # Handle boundary inventory: no bid at Q, no ask at -Q
    at_max_inventory = (inventory == Q)
    at_min_inventory = (inventory == -Q)
    
    bid_depths[at_max_inventory] = np.inf
    ask_depths[at_min_inventory] = np.inf
    
    # Clamp depths to valid range
    min_depth = -3.0 / k + 1e-9
    max_depth = 100.0 / k
    
    bid_depths = np.where(bid_depths < min_depth, min_depth, bid_depths)
    ask_depths = np.where(ask_depths < min_depth, min_depth, ask_depths)
    bid_depths = np.where(bid_depths > max_depth, max_depth, bid_depths)
    ask_depths = np.where(ask_depths > max_depth, max_depth, ask_depths)
    
    # Ensure no NaN or -inf
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths == -np.inf), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths == -np.inf), np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
