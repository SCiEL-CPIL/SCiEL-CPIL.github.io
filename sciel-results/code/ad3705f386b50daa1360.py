import numpy as np

def quote(t, inventory, mid, params):
    """
    Quotes bid and ask depths based on inventory risk management.
    
    Args:
        t: Current time.
        inventory: Array of inventory levels for each path.
        mid: Array of mid prices for each path.
        params: Dictionary containing 'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'.
        
    Returns:
        Tuple of two arrays (bid_depths, ask_depths).
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    Q = params['Q']
    
    n_paths = len(inventory)
    
    # Base depth: 1/k is a standard choice where exp(-k*depth) = 1/e
    d_base = 1.0 / k
    
    # Skew factor: proportional to risk penalty and time remaining
    # As time runs out, we skew less because we have less time to correct inventory?
    # Or do we skew more?
    # In Avellaneda-Stoikov, the reservation price shift is gamma * sigma^2 * q * (T-t).
    # So skew should be proportional to (T-t).
    time_remaining = 1.0 - t / T
    # Scale factor 1/k converts price units to depth units roughly.
    # Let's use a simple scaling.
    skew = (gamma * sigma**2 / k) * time_remaining
    
    # Calculate depths
    # d_bid = d_base + skew * q
    # d_ask = d_base - skew * q
    
    bid_depths = d_base + skew * inventory
    ask_depths = d_base - skew * inventory
    
    # Ensure depths are non-negative (minimum depth to avoid invalid quotes)
    # A very small depth is aggressive.
    min_depth = 1e-6
    
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    # Apply boundary conditions: no quoting on the side that would increase inventory beyond Q
    # If inventory is Q, we should not buy (bid).
    # If inventory is -Q, we should not sell (ask).
    
    # Identify paths at max inventory
    at_max_inv = (inventory == Q)
    # Identify paths at min inventory
    at_min_inv = (inventory == -Q)
    
    # Set bid depth to inf if at max inventory (don't buy)
    bid_depths[at_max_inv] = np.inf
    
    # Set ask depth to inf if at min inventory (don't sell)
    ask_depths[at_min_inv] = np.inf
    
    return bid_depths, ask_depths
