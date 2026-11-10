"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def quote(t, inventory, mid, params):
    """
    Inventory-aware market making policy using HJB-derived skew.
    
    Uses the approximation that the optimal quote skew is proportional to
    inventory and time remaining, with a constant derived from the risk
    aversion parameter gamma and volatility sigma.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    Q = params['Q']
    
    n_paths = len(inventory)
    inv = np.asarray(inventory, dtype=np.float64)
    
    # Time remaining
    t_rem = max(0.0, T - t)
    
    # Base depth (risk-neutral for single side)
    base_depth = 1.0 / k
    
    # Skew factor: gamma * sigma^2 * (T - t)
    # This comes from the first-order condition of the HJB equation
    # for a quadratic value function approximation.
    skew_factor = gamma * (sigma ** 2) * t_rem
    
    # Calculate depths
    # If q > 0 (long), we want to sell (ask).
    # Ask depth should be smaller (more attractive) -> base - skew*q
    # Bid depth should be larger (less attractive) -> base + skew*q
    bid_depths = base_depth + skew_factor * inv
    ask_depths = base_depth - skew_factor * inv
    
    # Enforce constraints: depth must be >= -3/k
    min_depth = -3.0 / k
    
    # Use np.maximum to clamp lower bound
    # If the calculated depth is too negative, we quote at the minimum valid depth
    # (most aggressive possible) to maximize the probability of reducing inventory.
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    # Ensure output is float arrays
    return bid_depths.astype(np.float64), ask_depths.astype(np.float64)
# EVOLVE-BLOCK-END
