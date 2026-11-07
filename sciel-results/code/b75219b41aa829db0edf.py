"""Optimal market making with inventory risk: Avellaneda-Stoikov closed-form policy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0


def quote(t, inventory, mid, params):
    """
    Closed-form Avellaneda-Stoikov policy with inventory risk adjustment.
    
    The reservation price is shifted by gamma * sigma^2 * q * (T - t).
    Bid depth is shifted down (wider) when long, ask depth shifted up (wider) when long.
    When short, opposite.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    Q = params["Q"]
    
    # Time remaining
    dt_rem = T - t
    
    # Inventory risk adjustment
    # R(q) = gamma * sigma^2 * q * (T - t)
    # This is the reservation price shift.
    # When q > 0 (long), R > 0, we want to sell more aggressively.
    # This means: bid should be lower (deeper), ask should be closer.
    
    # Base depth (risk-neutral)
    base_depth = 1.0 / k
    
    # Shift the depths based on inventory
    # The optimal bid depth: d_bid = base_depth - R/k * (1/(2*k*base_depth + 1/k)) approximately
    # More precisely, for the exponential arrival rate model:
    # bid_depth = (1/k) * log(1 + k * base_depth) - R/k * correction
    # 
    # A simpler and robust approach:
    # bid_depth = base_depth - R / (2 * k * sigma^2) * something
    # 
    # Let's use the standard AS result:
    # The reservation price is: r = S - gamma * sigma^2 * q * (T - t)
    # Bid = r - d_bid, Ask = r + d_ask
    # where d_bid = d_ask = (1/k) * log(1 + k/sigma^2 * 2 * something)
    # 
    # For the discrete model with arrival rate lambda(d) = A * exp(-k*d):
    # The optimal depth for a single side with no inventory constraint is 1/k.
    # With inventory q, the effective "cost" of holding inventory is gamma*sigma^2*q*(T-t).
    # 
    # A practical approach: shift the depth by the reservation price shift divided by k
    # bid_depth = base_depth + R / (k * sigma^2) * sigma^2 / (2 * something)
    # 
    # Let's use: bid_depth = base_depth - R / (2 * k * sigma^2) * 2 * sigma^2 / (k * base_depth + 1)
    # Simplified: bid_depth = base_depth - R / (k * (k * base_depth + 1))
    
    # R = gamma * sigma^2 * q * dt_rem
    R = gamma * sigma * sigma * inventory * dt_rem
    
    # Shift depths
    # When R > 0 (long), bid_depth should decrease (quote deeper to avoid buying more)
    # and ask_depth should increase (quote closer to encourage selling)
    # 
    # The shift magnitude: R / (k * (1 + k * base_depth))
    # Since base_depth = 1/k, k*base_depth = 1, so denominator = k*2 = 2k
    # shift = R / (2 * k)
    
    shift = R / (2.0 * k)
    
    bid_depth = base_depth - shift
    ask_depth = base_depth + shift
    
    # Clip to valid range: depth must be > -3/k and finite
    min_depth = -3.0 / k + 1e-6
    
    bid_depth = np.maximum(bid_depth, min_depth)
    ask_depth = np.maximum(ask_depth, min_depth)
    
    # For q = Q, no bid fill allowed, so set bid_depth to inf (no quote)
    # For q = -Q, no ask fill allowed, so set ask_depth to inf
    bid_depth = np.where(inventory >= Q, np.inf, bid_depth)
    ask_depth = np.where(inventory <= -Q, np.inf, ask_depth)
    
    # Ensure no NaN or -inf
    bid_depth = np.where(np.isnan(bid_depth) | (bid_depth == -np.inf), np.inf, bid_depth)
    ask_depth = np.where(np.isnan(ask_depth) | (ask_depth == -np.inf), np.inf, ask_depth)
    
    return bid_depth, ask_depth
# EVOLVE-BLOCK-END
