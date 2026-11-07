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
    # We work backwards from n = steps to n = 0
    
    # Value table: shape (steps+1, num_q)
    V = np.zeros((steps + 1, num_q), dtype=np.float64)
    
    # Depth tables: shape (steps, num_q)
    bid_depths = np.full((steps, num_q), np.inf, dtype=np.float64)
    ask_depths = np.full((steps, num_q), np.inf, dtype=np.float64)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        t_rem = T - n * dt
        t_next = (n + 1) * dt
        
        for qi in range(num_q):
            q = q_indices[qi]
            
            # Inventory penalty for current step
            penalty = 0.5 * gamma * sigma * sigma * q * q * dt
            
            # Try different bid and ask depths
            # We'll use a grid of depth values
            # For efficiency, we can optimize over depth analytically or use a small grid
            
            # Let's use a small grid of depth multipliers
            # d = base * factor
            base_d = _OPTIMAL_KD / k
            factors = np.array([0.5, 0.7, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0, 5.5])
            
            best_val = -np.inf
            best_bid = np.inf
            best_ask = np.inf
            
            for f_b in factors:
                d_bid = base_d * f_b
                if d_bid < -3.0 / k:
                    continue
                
                # Probability of bid fill
                p_bid = 1.0 - np.exp(-A * np.exp(-k * d_bid) * dt) if d_bid < np.inf else 0.0
                if q == Q:
                    p_bid = 0.0  # No bid fill at max inventory
                
                for f_a in factors:
                    d_ask = base_d * f_a
                    if d_ask < -3.0 / k:
                        continue
                    
                    # Probability of ask fill
                    p_ask = 1.0 - np.exp(-A * np.exp(-k * d_ask) * dt) if d_ask < np.inf else 0.0
                    if q == -Q:
                        p_ask = 0.0  # No ask fill at min inventory
                    
                    # Expected value if we quote these depths
                    # With prob p_bid: earn d_bid, inventory becomes q+1
                    # With prob p_ask: earn d_ask, inventory becomes q-1
                    # With prob (1-p_bid)(1-p_ask): no fill, inventory stays q
                    # Mid moves, but that doesn't directly affect value in this simplified model
                    # (the value function V already accounts for future mid movements)
                    
                    # Actually, we need to be more careful. The mid moves, but our value function
                    # is defined in terms of inventory only, assuming the mid is "normalized" away.
                    # In the Avellaneda-Stoikov framework, we can work in excess return space.
                    
                    # Expected immediate reward
                    reward = d_bid * p_bid + d_ask * p_ask - penalty
                    
                    # Expected future value
                    # If bid fills: q -> q+1
                    # If ask fills: q -> q-1
                    # If both fill: q -> q (bid then ask, or ask then bid)
                    # If neither: q -> q
                    
                    p_both = p_bid * p_ask
                    p_bid_only = p_bid * (1 - p_ask)
                    p_ask_only = p_ask * (1 - p_bid)
                    p_neither = (1 - p_bid) * (1 - p_ask)
                    
                    # Next inventory states
                    q_next_bid = min(q + 1, Q)
                    q_next_ask = max(q - 1, -Q)
                    q_next_both = q  # bid then ask, or ask then bid, net zero
                    q_next_neither = q
                    
                    # Convert to indices
                    qi_bid = q_next_bid + Q
                    qi_ask = q_next_ask + Q
                    qi_both = q_next_both + Q
                    qi_neither = q_next_neither + Q
                    
                    # V at next time step
                    v_bid = V[n + 1, qi_bid]
                    v_ask = V[n + 1, qi_ask]
                    v_both = V[n + 1, qi_both]
                    v_neither = V[n + 1, qi_neither]
                    
                    expected_future = (p_bid_only * v_bid + 
                                       p_ask_only * v_ask + 
                                       p_both * v_both + 
                                       p_neither * v_neither)
                    
                    total_val = reward + expected_future
                    
                    if total_val > best_val:
                        best_val = total_val
                        best_bid = d_bid
                        best_ask = d_ask
            
            V[n, qi] = best_val
            bid_depths[n, qi] = best_bid
            ask_depths[n, qi] = best_ask
    
    _dp_cache[cache_key] = (bid_depths, ask_depths)
    return bid_depths, ask_depths


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
    if elapsed > 320.0:  # 330s limit minus 10s margin
        # Fallback to simple log-depth skew
        base_depth = _OPTIMAL_KD / k
        inv_float = inventory.astype(np.float64)
        t_rem = T - t
        x = gamma * sigma * sigma * t_rem * inv_float
        with np.errstate(divide='ignore', invalid='ignore'):
            denom_bid = 1.0 - x
            denom_ask = 1.0 + x
            raw_bid = np.log(_BASE_EXP / denom_bid) / k
            raw_ask = np.log(_BASE_EXP / denom_ask) / k
        bid_depths = np.where(denom_bid <= 0, np.inf, raw_bid)
        ask_depths = np.where(denom_ask <= 0, np.inf, raw_ask)
        min_depth = -3.0 / k + 1e-9
        max_depth = 100.0 / k
        bid_depths = np.clip(bid_depths, min_depth, max_depth)
        ask_depths = np.clip(ask_depths, min_depth, max_depth)
        bid_depths[inventory == Q] = np.inf
        ask_depths[inventory == -Q] = np.inf
        bid_depths = np.where(np.isnan(bid_depths) | (bid_depths == -np.inf), np.inf, bid_depths)
        ask_depths = np.where(np.isnan(ask_depths) | (ask_depths == -np.inf), np.inf, ask_depths)
        return bid_depths, ask_depths
    
    # Get or compute optimal policy
    bid_table, ask_table = _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q)
    
    # Determine current time step
    n = int(round(t / dt))
    n = np.clip(n, 0, steps - 1)
    
    # Map inventory to index
    q_indices = inventory + Q  # Convert from [-Q, Q] to [0, 2Q]
    q_indices = np.clip(q_indices, 0, 2 * Q)
    
    # Look up depths
    bid_depths = bid_table[n, q_indices]
    ask_depths = ask_table[n, q_indices]
    
    # Handle boundary inventory: no bid at Q, no ask at -Q
    at_max_inventory = (inventory == Q)
    at_min_inventory = (inventory == -Q)
    
    bid_depths[at_max_inventory] = np.inf
    ask_depths[at_min_inventory] = np.inf
    
    # Ensure valid values
    min_depth = -3.0 / k + 1e-9
    max_depth = 100.0 / k
    
    # Only clip finite values
    finite_bid = np.isfinite(bid_depths)
    finite_ask = np.isfinite(ask_depths)
    
    bid_depths[finite_bid] = np.clip(bid_depths[finite_bid], min_depth, max_depth)
    ask_depths[finite_ask] = np.clip(ask_depths[finite_ask], min_depth, max_depth)
    
    # Ensure no NaN or -inf
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths == -np.inf), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths == -np.inf), np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
