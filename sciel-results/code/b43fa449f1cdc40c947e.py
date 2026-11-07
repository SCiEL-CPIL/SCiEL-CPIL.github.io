"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds of safety margin

def quote(t, inventory, mid, params):
    """
    Inventory-aware market making strategy with deadline guard.
    
    Uses a theoretically derived skew based on the Avellaneda-Stoikov style
    inventory penalty to adjust bid and ask depths.
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
        depth = np.full(N, 1.0 / k)
        return depth, depth.copy()
    
    # Time remaining
    t_rem = T - t
    
    # Base depth: risk-neutral quote depth
    base_depth = 1.0 / k
    
    # Compute the inventory skew term.
    # The optimal additive skew to the half-spread is proportional to 
    # gamma * sigma^2 * t_rem * inventory.
    # We convert this additive term to a multiplicative factor on base_depth
    # by multiplying by k (since base_depth = 1/k).
    # skew_mult = k * (gamma * sigma^2 * t_rem * inventory)
    
    inv_float = inventory.astype(np.float64)
    risk_factor = gamma * sigma * sigma * t_rem
    skew_mult = k * risk_factor * inv_float
    
    # Compute bid and ask depths
    # bid_depth = base_depth * (1 + skew_mult)
    # ask_depth = base_depth * (1 - skew_mult)
    
    bid_depths = base_depth * (1.0 + skew_mult)
    ask_depths = base_depth * (1.0 - skew_mult)
    
    # Clip to valid range: must be > -3/k and finite
    # Also, depths should ideally be positive to be "below/above", 
    # but the constraint allows down to -3/k.
    min_depth = -3.0 / k
    # Upper bound: very large depth means no fill. 
    # A reasonable upper bound to prevent numerical issues or excessive widening.
    # If depth is too large, prob ~ 0. 
    # Let's cap at 10.0 / k or similar, but inf is also allowed.
    # However, returning inf is valid. But large finite is safer for some evaluators?
    # The prompt says np.inf means no quote.
    # Let's allow large values but clip to a reasonable max to avoid overflow in exp if used elsewhere?
    # The model uses exp(-k * depth). If depth is large positive, exp is 0.
    # If depth is large negative, exp is huge. But we clip to -3/k.
    # So max_depth can be large.
    max_depth = 100.0 / k
    
    bid_depths = np.clip(bid_depths, min_depth, max_depth)
    ask_depths = np.clip(ask_depths, min_depth, max_depth)
    
    # Handle boundary inventory: no bid at Q, no ask at -Q
    at_max_inventory = (inventory == Q)
    at_min_inventory = (inventory == -Q)
    
    # Set bid to inf where we're at max inventory (can't take more bids)
    bid_depths[at_max_inventory] = np.inf
    
    # Set ask to inf where we're at min inventory (can't take more asks)
    ask_depths[at_min_inventory] = np.inf
    
    # Ensure no NaN or -inf
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths == -np.inf), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths == -np.inf), np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
