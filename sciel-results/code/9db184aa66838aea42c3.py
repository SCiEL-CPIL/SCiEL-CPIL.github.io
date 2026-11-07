"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_start_time = None

# Precompute optimal base depth factor
# Maximize d * (1 - exp(-k*d)) -> let y = k*d, maximize y*(1-exp(-y))
# Derivative: (1-exp(-y)) - y*exp(-y) = 0 => 1 - exp(-y)(1+y) = 0
# y* = 1.5936242601192586
_OPTIMAL_KD = 1.5936242601192586
_BASE_EXP = np.exp(_OPTIMAL_KD)  # ~4.921

def quote(t, inventory, mid, params):
    """
    Inventory-aware market making using the log-depth formula derived from
    the quadratic value function approximation (Avellaneda-Stoikov style).
    
    d_bid = (1/k) * ln(exp(y*) / (1 - x))
    d_ask = (1/k) * ln(exp(y*) / (1 + x))
    where x = gamma * sigma^2 * t_rem * q
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are very close to the limit, return a safe fallback
    if time.time() - _start_time > 320:
        k = params['k']
        base = _OPTIMAL_KD / k
        return np.full(len(inventory), base), np.full(len(inventory), base)
    
    N = len(inventory)
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Time remaining
    t_rem = T - t
    
    # Inventory as float
    inv_float = inventory.astype(np.float64)
    
    # Compute the skew argument
    # x = gamma * sigma^2 * t_rem * q
    x = gamma * sigma * sigma * t_rem * inv_float
    
    # Compute depths using log formula:
    # d_bid = (1/k) * ln( _BASE_EXP / (1 - x) )
    # d_ask = (1/k) * ln( _BASE_EXP / (1 + x) )
    
    with np.errstate(divide='ignore', invalid='ignore'):
        denom_bid = 1.0 - x
        denom_ask = 1.0 + x
        
        raw_bid = np.log(_BASE_EXP / denom_bid) / k
        raw_ask = np.log(_BASE_EXP / denom_ask) / k
    
    # Fix invalid values: if denom <= 0, depth should be inf (no quote)
    bid_depths = np.where(denom_bid <= 0, np.inf, raw_bid)
    ask_depths = np.where(denom_ask <= 0, np.inf, raw_ask)
    
    # Clamp to valid range: depth > -3/k
    min_depth = -3.0 / k + 1e-9
    max_depth = 100.0 / k
    
    # Only clamp finite values; leave inf as inf
    finite_bid = np.isfinite(bid_depths)
    finite_ask = np.isfinite(ask_depths)
    
    bid_depths[finite_bid] = np.clip(bid_depths[finite_bid], min_depth, max_depth)
    ask_depths[finite_ask] = np.clip(ask_depths[finite_ask], min_depth, max_depth)
    
    # Handle boundary inventory: no bid at Q, no ask at -Q
    at_max_inventory = (inventory == Q)
    at_min_inventory = (inventory == -Q)
    
    bid_depths[at_max_inventory] = np.inf
    ask_depths[at_min_inventory] = np.inf
    
    # Ensure no NaN or -inf
    bad_bid = np.isnan(bid_depths) | (bid_depths == -np.inf) | (bid_depths <= -3.0/k)
    bad_ask = np.isnan(ask_depths) | (ask_depths == -np.inf) | (ask_depths <= -3.0/k)
    
    if np.any(bad_bid):
        bid_depths[bad_bid] = min_depth
    if np.any(bad_ask):
        ask_depths[bad_ask] = min_depth
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
