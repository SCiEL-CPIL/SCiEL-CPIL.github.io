"""Optimal market making with inventory risk - analytic HJB-based quote."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute nothing; the formula is closed-form per call.

def quote(t, inventory, mid, params):
    """
    Quote both sides using the analytic first-order-condition optimal depth.
    
    The HJB first-order condition gives:
      bid_depth  = (1 + gamma * sigma^2 * q * tau / k) / k   [tighter when q>0]
      ask_depth  = (1 - gamma * sigma^2 * q * tau / k) / k   [wider when q>0]
    
    where tau = T - t is time remaining.
    Clipped to valid range [0, 3/k).
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    Q = params["Q"]
    
    n = len(inventory)
    tau = T - t  # time remaining
    
    # Inventory as float array
    q = inventory.astype(np.float64)
    
    # The skew term: how much inventory risk shifts the optimal depth
    # From HJB: marginal value of inventory ~ gamma * sigma^2 * q * tau
    # FOC: 1 - k*d = delta_V  =>  d = (1 - delta_V) / k
    # For bid (reduces q by 1): delta_V = V(q-1) - V(q) ~ -dV/dq ~ -gamma*sigma^2*q*tau
    #   So bid_depth = (1 + gamma*sigma^2*q*tau) / k
    # For ask (increases q by 1): delta_V = V(q+1) - V(q) ~ dV/dq ~ gamma*sigma^2*q*tau
    #   So ask_depth = (1 - gamma*sigma^2*q*tau) / k
    
    skew = gamma * sigma * sigma * q * tau
    
    bid_depth = (1.0 + skew) / k
    ask_depth = (1.0 - skew) / k
    
    # Clamp to valid range: must be >= 0 and < 3/k
    max_depth = 3.0 / k
    min_depth = 0.0
    
    bid_depth = np.clip(bid_depth, min_depth, max_depth - 1e-12)
    ask_depth = np.clip(ask_depth, min_depth, max_depth - 1e-12)
    
    # At inventory bounds: no bid at q=Q, no ask at q=-Q
    # When q == Q, set bid to inf (no quote on bid side)
    # When q == -Q, set ask to inf (no quote on ask side)
    at_max_inv = (inventory >= Q)
    at_min_inv = (inventory <= -Q)
    
    bid_depth[at_max_inv] = np.inf
    ask_depth[at_min_inv] = np.inf
    
    # Ensure no NaN or negative values
    bid_depth = np.where(np.isnan(bid_depth) | (bid_depth < 0), np.inf, bid_depth)
    ask_depth = np.where(np.isnan(ask_depth) | (ask_depth < 0), np.inf, ask_depth)
    
    return bid_depth, ask_depth
# EVOLVE-BLOCK-END
