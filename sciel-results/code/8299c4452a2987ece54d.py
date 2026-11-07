"""Pre-computed HJB lookup table for optimal market making with inventory risk."""
import numpy as np
from functools import lru_cache

# Global cache for pre-computed lookup tables
_lookup_cache = {}


def _build_lookup_table(sigma, A, k, gamma, Q=30, steps=1000):
    """
    Build a pre-computed lookup table using coarse-grained HJB backward induction.
    
    Returns a 2D array of shape (steps, 2*Q+1) for bid and ask depth indices,
    and the depth grid used.
    """
    dt = 1.0 / steps
    
    # Reduced depth grid - use a logarithmic or linear grid
    # Depths range from small values to about 1/k * 2
    max_depth = 2.0 / k
    n_depths = 50
    depth_grid = np.linspace(0.01, max_depth, n_depths)
    
    # Inventory range
    inv_min = -Q
    inv_max = Q
    n_inv = 2 * Q + 1
    
    # Value function: V[step, inv_idx]
    # We'll store bid and ask depth indices for each (step, inventory) pair
    bid_depths_table = np.zeros((steps, n_inv), dtype=np.float64)
    ask_depths_table = np.zeros((steps, n_inv), dtype=np.float64)
    
    # Terminal condition: no value
    V = np.zeros(n_inv)
    
    # Backward induction
    for t in range(steps - 1, -1, -1):
        V_new = np.zeros(n_inv)
        
        for inv_idx in range(n_inv):
            q = inv_min + inv_idx
            
            # If at inventory boundary, can't quote on that side
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            best_value = -np.inf
            best_bid_depth = np.inf
            best_ask_depth = np.inf
            
            # Try all combinations of bid and ask depths
            # To speed up, we can use a coarser search
            # But for now, let's do a reasonable grid
            
            # Optimization: limit depth options
            # We know optimal depths are typically around 1/k
            # Let's use a subset of the depth grid
            
            bid_options = depth_grid if can_bid else np.array([np.inf])
            ask_options = depth_grid if can_ask else np.array([np.inf])
            
            # For performance, use a coarser grid for the search
            # Then refine
            bid_coarse = bid_options[::2]
            ask_coarse = ask_options[::2]
            
            for bid_d in bid_coarse:
                if np.isinf(bid_d):
                    p_bid = 0.0
                else:
                    p_bid = 1.0 - np.exp(-A * np.exp(-k * bid_d) * dt)
                
                for ask_d in ask_coarse:
                    if np.isinf(ask_d):
                        p_ask = 0.0
                    else:
                        p_ask = 1.0 - np.exp(-A * np.exp(-k * ask_d) * dt)
                    
                    # Compute expected value
                    # Four outcomes: no fill, bid fill, ask fill, both fills
                    p_none = (1 - p_bid) * (1 - p_ask)
                    p_bid_fill = p_bid * (1 - p_ask)
                    p_ask_fill = (1 - p_bid) * p_ask
                    p_both = p_bid * p_ask
                    
                    # Inventory after fills
                    # Bid fill: inventory increases by 1 (we buy, so we hold more)
                    # Ask fill: inventory decreases by 1 (we sell)
                    # Wait, let me reconsider: in market making, when you bid and get filled,
                    # you're buying the asset, so your inventory increases
                    # When you ask and get filled, you're selling, so inventory decreases
                    
                    new_inv_bid = q + 1 if q < Q else q
                    new_inv_ask = q - 1 if q > -Q else q
                    new_inv_both = q  # +1 and -1 cancel out
                    
                    inv_idx_bid = new_inv_bid - inv_min
                    inv_idx_ask = new_inv_ask - inv_min
                    inv_idx_both = new_inv_both - inv_min
                    
                    # Rewards
                    # Bid fill: you earn bid_depth (you bought at a lower price, so you gain the depth)
                    # Actually, the objective says "depths earned", so bid fill earns bid_depth
                    # Ask fill earns ask_depth
                    # Inventory penalty: gamma/2 * sigma^2 * q^2 * dt where q is inventory BEFORE fills
                    
                    reward_none = 0.0
                    reward_bid = bid_d if np.isfinite(bid_d) else 0.0
                    reward_ask = ask_d if np.isfinite(ask_d) else 0.0
                    reward_both = (bid_d if np.isfinite(bid_d) else 0.0) + (ask_d if np.isfinite(ask_d) else 0.0)
                    
                    inv_penalty = (gamma / 2.0) * sigma**2 * q**2 * dt
                    
                    # Expected value
                    expected_value = (
                        p_none * (V[inv_idx] + reward_none) +
                        p_bid_fill * (V[inv_idx_bid] + reward_bid) +
                        p_ask_fill * (V[inv_idx_ask] + reward_ask) +
                        p_both * (V[inv_idx_both] + reward_both)
                    ) - inv_penalty
                    
                    if expected_value > best_value:
                        best_value = expected_value
                        best_bid_depth = bid_d
                        best_ask_depth = ask_d
            
            V_new[inv_idx] = best_value
            bid_depths_table[t, inv_idx] = best_bid_depth
            ask_depths_table[t, inv_idx] = best_ask_depth
        
        V = V_new
    
    return bid_depths_table, ask_depths_table


def _get_lookup_table(params):
    """Get or build the lookup table for given parameters."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params.get('Q', 30)
    steps = params.get('steps', 1000)
    
    key = (sigma, A, k, gamma, Q, steps)
    
    if key not in _lookup_cache:
        _lookup_cache[key] = _build_lookup_table(sigma, A, k, gamma, Q, steps)
    
    return _lookup_cache[key]


def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths for each path based on pre-computed HJB lookup table.
    
    Returns (bid_depths, ask_depths) as float arrays of shape (4000,).
    """
    Q = params.get('Q', 30)
    steps = params.get('steps', 1000)
    
    # Determine current step
    step = int(round(t / (1.0 / steps)))
    step = max(0, min(steps - 1, step))
    
    bid_depths_table, ask_depths_table = _get_lookup_table(params)
    
    # Map inventory to index
    inv_min = -Q
    inv_idx = inventory - inv_min
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Look up depths for each path
    bid_depths = bid_depths_table[step, inv_idx]
    ask_depths = ask_depths_table[step, inv_idx]
    
    # Ensure values are valid (no negative depths, no NaN)
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > 0), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > 0), ask_depths, np.inf)
    
    return bid_depths, ask_depths
