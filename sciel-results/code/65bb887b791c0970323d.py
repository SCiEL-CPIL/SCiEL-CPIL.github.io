import numpy as np

def quote(t, inventory, mid, params):
    """
    Quote depths based on inventory skew.
    
    Args:
        t: time
        inventory: array of shape (4000,)
        mid: array of shape (4000,)
        params: dict with 'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'
    
    Returns:
        tuple (bid_depths, ask_depths)
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    
    # Calculate skew parameter
    # Higher risk (gamma, sigma) or lower fill intensity (A, k) should lead to more skew
    # Dimensional analysis:
    # Depth has units of price.
    # q is inventory.
    # lambda * q should be dimensionless.
    # lambda has units of 1/inventory.
    # gamma * sigma^2 has units of price^2 / time * (1/time)? No.
    # Objective term: gamma * sigma^2 * q^2 * dt.
    # gamma is likely a risk aversion coefficient.
    
    # Let's use a heuristic scaling.
    # The "cost" of inventory is ~ gamma * sigma^2 * q.
    # The "benefit" of a fill is ~ depth.
    # We want to balance these.
    
    # lambda = (gamma * sigma^2) / (A * k)
    # Let's check magnitudes again.
    # Regime 1: 0.05 * 4 / (140 * 1.5) = 0.2 / 210 = 0.00095
    # Regime 2: 0.5 * 4 / (140 * 1.5) = 2.0 / 210 = 0.0095
    # Regime 3: 0.2 * 2.25 / (50 * 0.75) = 0.45 / 37.5 = 0.012
    
    # With Q=30:
    # Reg 1: exp(0.00095 * 30) = exp(0.028) ~ 1.03. Very small skew.
    # Reg 2: exp(0.0095 * 30) = exp(0.285) ~ 1.33. Moderate skew.
    # Reg 3: exp(0.012 * 30) = exp(0.36) ~ 1.43. Moderate skew.
    
    # This seems reasonable.
    
    lambda_skew = (gamma * sigma**2) / (A * k)
    
    # Ensure lambda is not too large to cause overflow or extreme quotes
    # Cap lambda to prevent exp(lambda*q) from being too huge
    max_lambda = 0.1 # heuristic cap
    lambda_skew = min(lambda_skew, max_lambda)
    
    # Calculate depths
    # d_bid = (1/k) * exp(-lambda * q)
    # d_ask = (1/k) * exp(lambda * q)
    
    # Clip inventory to valid range if necessary, though it should be in [-Q, Q]
    q = np.clip(inventory, -params['Q'], params['Q'])
    
    base_depth = 1.0 / k
    
    # Vectorized calculation
    # Use np.exp for speed
    # Handle potential overflow by clipping the exponent
    max_exp_arg = 50.0 # exp(50) is ~5e21, which is fine for float64, but depth might be huge
    # Actually, if depth is huge, prob is 0.
    # We can just let it be.
    
    exp_bid = np.exp(-lambda_skew * q)
    exp_ask = np.exp(lambda_skew * q)
    
    bid_depths = base_depth * exp_bid
    ask_depths = base_depth * exp_ask
    
    # Ensure depths are positive and not NaN/Inf
    # If exp overflows to inf, depth is inf, which means no quote. This is valid.
    # If exp underflows to 0, depth is 0.
    # Constraint: Values below -3/k are invalid.
    # Since base_depth > 0 and exp > 0, depths are > 0. So they are > -3/k.
    
    # Check for any invalid values just in case
    if not np.all(np.isfinite(bid_depths)):
        # Replace inf with a large number? No, inf means no quote.
        # But if NaN appears, we must fix it.
        bid_depths = np.nan_to_num(bid_depths, nan=base_depth, posinf=np.inf, neginf=base_depth)
        ask_depths = np.nan_to_num(ask_depths, nan=base_depth, posinf=np.inf, neginf=base_depth)
        
    return bid_depths, ask_depths
