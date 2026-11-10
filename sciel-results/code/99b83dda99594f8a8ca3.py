<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def quote(t, inventory, mid, params):
    """Seed: quote both sides at the risk-neutral depth 1/k, ignoring inventory."""
    depth = np.full(len(inventory), 1.0 / params["k"])
    return depth, depth.copy()
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def quote(t, inventory, mid, params):
    """
    Market making quote with inventory risk adjustment.
    
    Uses a risk-averse market making strategy that adjusts quote depths
    based on current inventory to manage risk.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    n = len(inventory)
    
    # Base depth: 1/k is the risk-neutral point where marginal fill cost = marginal value
    base_depth = 1.0 / k
    
    # Inventory risk adjustment: shift quotes to reduce inventory
    # When long (q > 0), widen bid and narrow ask to sell more
    # When short (q < 0), narrow bid and widen ask to buy more
    
    # Risk adjustment parameter based on gamma, sigma, and time remaining
    # The optimal skew is proportional to gamma * sigma^2 * q * remaining_time
    dt = T / steps
    t_remaining = T - t
    
    # Adjustment factor: how much to skew based on inventory
    # This comes from balancing the inventory cost gradient with fill probability
    # The cost gradient is gamma * sigma^2 * q
    # We want to shift depth by an amount that makes marginal value = marginal cost
    
    # A reasonable heuristic: adjustment = c * gamma * sigma^2 * q / (k * A * exp(-k * base_depth))
    # But let's use a simpler form that's known to work well
    
    # The HWHM model suggests: reservation price shifts by gamma * sigma^2 * q * t_remaining
    # For depths, we can use: bid_depth = base_depth - alpha * q, ask_depth = base_depth + alpha * q
    
    # Choose alpha based on the problem parameters
    # alpha should be small enough to keep depths positive
    # alpha = gamma * sigma^2 * t_remaining / (k * A * exp(-k * base_depth))
    
    exp_term = np.exp(-k * base_depth)
    alpha = gamma * sigma**2 * t_remaining / (k * A * exp_term)
    
    # Clamp alpha to reasonable values
    alpha = np.clip(alpha, 0.0, 0.5)
    
    # Compute adjusted depths
    q = inventory.astype(np.float64)
    
    bid_depth = base_depth - alpha * q
    ask_depth = base_depth + alpha * q
    
    # Ensure depths are positive and not too large
    min_depth = 1e-6
    max_depth = 10.0 / k
    
    bid_depth = np.clip(bid_depth, min_depth, max_depth)
    ask_depth = np.clip(ask_depth, min_depth, max_depth)
    
    # At boundary inventory, don't quote the side that would increase inventory further
    # No bid fill at q = Q (can't buy more)
    # No ask fill at q = -Q (can't sell more)
    
    # Set bid to inf when at Q (can't buy)
    bid_depth[inventory >= Q] = np.inf
    # Set ask to inf when at -Q (can't sell)
    ask_depth[inventory <= -Q] = np.inf
    
    return bid_depth, ask_depth
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
BET: {mechanism: "Inventory-aware quote skewing reduces expected inventory risk cost by shifting quotes to favor reducing position, which should improve the objective over the inventory-ignoring seed", q: 0.7, delta: [0.01, 0.15]}
