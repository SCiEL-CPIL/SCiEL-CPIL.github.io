"""
Rewritten program using an Analytic Hedging Heuristic with time-decay.
The policy skews the bid and ask depths based on current inventory to manage risk,
approximating the first-order optimal control solution. Skew is reduced as time
approaches the horizon to avoid excessive risk aversion when there is little time
left to adjust inventory.
"""
# EVOLVE-BLOCK-START
import numpy as np


def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths based on inventory risk with time-decay.
    
    Args:
        t: Current time step (float).
        inventory: Array of current inventories (int array).
        mid: Array of mid prices (float array).
        params: Dict with 'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'.
        
    Returns:
        Tuple of (bid_depths, ask_depths) as float arrays.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    Q = params['Q']
    
    inv = inventory.astype(np.float64)
    
    # Base depth: risk-neutral depth where fill probability is balanced
    base_depth = 1.0 / k
    
    # Calculate inventory skew with time decay
    # As time approaches the horizon, we reduce the skew to avoid
    # being too conservative when there's little time to adjust.
    # The decay factor goes from 1 at t=0 to a small value at t=T.
    remaining_fraction = max(0.0, 1.0 - t / T)
    # Use a smooth decay: 0.2 + 0.8 * remaining_fraction
    # This keeps some skew even near the end but reduces it significantly.
    decay_factor = 0.2 + 0.8 * remaining_fraction
    
    # The term (gamma * sigma^2) / (A * k) represents the strength of the risk aversion
    # relative to the fill intensity and depth sensitivity.
    skew_factor = (gamma * sigma**2) / (A * k) * decay_factor
    skew = skew_factor * inv
    
    # Compute raw depths
    bid_depths = base_depth * (1.0 + skew)
    ask_depths = base_depth * (1.0 - skew)
    
    # Ensure depths are positive (minimum depth epsilon)
    min_depth = 1e-6
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    # Handle inventory bounds:
    # No bid fill at q = Q (we are max long, don't buy more)
    # No ask fill at q = -Q (we are max short, don't sell more)
    bid_depths[inv >= Q] = np.inf
    ask_depths[inv <= -Q] = np.inf
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
