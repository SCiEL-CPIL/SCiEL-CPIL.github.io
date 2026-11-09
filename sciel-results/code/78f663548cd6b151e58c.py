import numpy as np

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    
    # Time remaining
    tau = T - t
    
    # Inventory skew term
    # The reservation price shift is q * gamma * sigma^2 * tau
    # The optimal half-spread is (1/k) * ln(1 + k / (gamma * sigma^2))
    
    # Calculate half-spread
    # Handle case where gamma * sigma^2 is very small or zero
    g_s2 = gamma * sigma * sigma
    if g_s2 < 1e-9:
        half_spread = 1.0 / k
    else:
        half_spread = (1.0 / k) * np.log1p(k / g_s2)
        
    # Calculate skew
    skew = inventory * g_s2 * tau
    
    # Depths
    # bid_depth = mid - bid_price
    # bid_price = r - half_spread = (mid - skew) - half_spread
    # bid_depth = mid - (mid - skew - half_spread) = skew + half_spread
    # 
    # ask_depth = ask_price - mid
    # ask_price = r + half_spread = (mid - skew) + half_spread
    # ask_depth = (mid - skew + half_spread) - mid = half_spread - skew
    
    bid_depths = skew + half_spread
    ask_depths = half_spread - skew
    
    # Clamp depths to be non-negative
    # If depth is negative, it means the optimal quote is on the other side of mid,
    # which implies a crossing quote. In market making, we usually don't cross.
    # Setting to 0 means no quote on that side.
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    return bid_depths, ask_depths
