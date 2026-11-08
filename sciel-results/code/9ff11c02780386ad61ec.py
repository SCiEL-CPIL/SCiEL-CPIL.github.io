"""Optimal market making with inventory risk using state-dependent depth adjustment."""
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill


def quote(t, inventory, mid, params):
    """State-dependent quoting: adjust depths based on inventory to manage risk."""
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Return safe default
        k = params["k"]
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()
    
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    Q = params["Q"]
    
    n = len(inventory)
    q = inventory.astype(np.float64)
    
    # Time remaining
    t_remaining = max(T - t, 0.0)
    
    # Base risk-neutral depth
    base_depth = 1.0 / k
    
    # Inventory adjustment: proportional to gamma * sigma^2 * q * t_remaining
    # This is the shadow price of inventory
    delta = (gamma * sigma**2 * q * t_remaining) / (2.0 * k)
    
    # Adjusted depths
    bid_depths = base_depth + delta
    ask_depths = base_depth - delta
    
    # Ensure depths are positive (minimum small positive value)
    min_depth = 1e-8
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    # Handle boundary conditions: no bid at q=Q, no ask at q=-Q
    bid_depths[q >= Q] = np.inf
    ask_depths[q <= -Q] = np.inf
    
    return bid_depths, ask_depths
