"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def quote(t, inventory, mid, params):
    """
    Optimal market making with inventory risk.
    
    Derivation: The HJB equation gives optimal depths that adjust based on
    inventory to hedge risk. For inventory q, the marginal value of inventory
    is approximately -gamma*sigma^2*q*(T-t), leading to:
      bid_depth  = (1/k) + gamma*sigma^2*q*(T-t)*dt   (wider when q>0)
      ask_depth  = (1/k) - gamma*sigma^2*q*(T-t)*dt   (tighter when q>0)
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    dt = T / steps
    n = int(round(t / dt)) if dt > 0 else 0
    time_remaining = max(T - t, 0.0)
    
    # Base depth (risk-neutral)
    base_depth = 1.0 / k
    
    # Inventory adjustment: gamma * sigma^2 * q * time_remaining * dt
    # This is the marginal value of inventory change
    inv = inventory.astype(np.float64)
    adjustment = gamma * (sigma ** 2) * inv * time_remaining * dt
    
    # Bid: wider when q > 0 (to avoid more inventory)
    bid_depths = base_depth + adjustment
    # Ask: tighter when q > 0 (to reduce inventory)
    ask_depths = base_depth - adjustment
    
    # Clip to valid range: must be > 0 and < 3/k
    min_depth = 1e-10
    max_depth = 3.0 / k - 1e-10
    
    bid_depths = np.clip(bid_depths, min_depth, max_depth)
    ask_depths = np.clip(ask_depths, min_depth, max_depth)
    
    # No bid fill at q = Q, no ask fill at q = -Q
    # At q = Q, we should not quote bid (would exceed max inventory)
    # At q = -Q, we should not quote ask
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
