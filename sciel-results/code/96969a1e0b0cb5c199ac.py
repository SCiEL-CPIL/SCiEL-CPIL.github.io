"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds of safety margin

# Precompute optimal base depth factor
# Maximize d * (1 - exp(-k*d)) -> k*d is root of 1 - exp(-y) = y*exp(-y)
# y* = 1.59362426...
_OPTIMAL_KD = 1.59362426
_BASE_EXP = np.exp(_OPTIMAL_KD)  # ~4.92

def quote(t, inventory, mid, params):
    """
    Inventory-aware market making strategy with deadline guard.
    
    Uses a log-depth skew based on the Avellaneda-Stoikov optimal policy
    approximation with quadratic value function.
    """
    N = len(inventory)
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Deadline guard: if we're close to the limit, return a simple safe quote
    elapsed = time.time() - _start_time
    if elapsed > 325.0:  # 330s limit minus 5s margin
        # Fallback to simple linear skew with optimal base
        base_depth = _OPTIMAL_KD / k
        inv_float = inventory.astype(np.float64)
        t_rem = T - t
        skew = gamma * sigma * sigma * t_rem * inv_float
        bid_depths = base_depth + skew
        ask_depths = base_depth - skew
        bid_depths = np.clip(bid_depths, -3.0/k, 100.0/k)
        ask_depths = np.clip(ask_depths, -3.0/k, 100.0/k)
        bid_depths[inventory == Q] = np.inf
        ask_depths[inventory == -Q] = np.inf
        return bid_depths, ask_depths
    
    # Time remaining
    t_rem = T - t
    
    # Inventory as float
    inv_float = inventory.astype(np.float64)
    
    # Compute the skew argument
    # x = gamma * sigma^2 * t_rem * q
    x = gamma * sigma * sigma * t_rem * inv_float
    
    # Compute depths using log formula:
    # d = (1/k) * ln( (1+c) / (1 +/- x) )
    # where 1+c = exp(k * d*) = _BASE_EXP
    # d_bid = (1/k) * ln( _BASE_EXP / (1 - x) )
    # d_ask = (1/k) * ln( _BASE_EXP / (1 + x) )
    
    # Handle division by zero or negative arguments in log
    # If 1 - x <= 0, d_bid -> inf (no quote, or very wide)
    # If 1 + x <= 0, d_ask -> inf
    
    # We use np.errstate to suppress warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        denom_bid = 1.0 - x
        denom_ask = 1.0 + x
        
        # If denom is <= 0, set to small positive to get large positive depth, 
        # or handle separately.
        # Actually, if 1-x <= 0, the optimal strategy is to not quote bid (inf).
        # But the formula gives -inf or nan.
        # We will compute raw values and then fix.
        
        raw_bid = np.log(_BASE_EXP / denom_bid) / k
        raw_ask = np.log(_BASE_EXP / denom_ask) / k
        
    # Fix invalid values
    # If denom <= 0, depth should be inf (or very large)
    # If denom > 0, depth is finite.
    
    # For bid: if 1-x <= 0, x >= 1. Depth should be inf.
    # For ask: if 1+x <= 0, x <= -1. Depth should be inf.
    
    # Check for non-finite or negative results from log (which shouldn't happen if denom>0 and _BASE_EXP>0)
    # But if denom < 0, log of negative is nan.
    
    # Set to inf where denom <= 0
    bid_depths = np.where(denom_bid <= 0, np.inf, raw_bid)
    ask_depths = np.where(denom_ask <= 0, np.inf, raw_ask)
    
    # Also, if the computed depth is negative (possible if _BASE_EXP < denom), 
    # it means we should quote tighter than optimal base?
    # But depth must be > -3/k.
    # The log formula ensures depth > 0 if denom < _BASE_EXP.
    # If denom > _BASE_EXP, depth < 0.
    # This happens if x is negative for bid (q < 0).
    # If q < 0 (short), we want to buy.
    # d_bid should be smaller (tighter).
    # d_ask should be larger (wider).
    # Formula:
    # q < 0 -> x < 0.
    # denom_bid = 1 - x = 1 + |x| > 1.
    # If |x| is large, denom_bid can be > _BASE_EXP.
    # Then log(_BASE_EXP / denom_bid) < 0.
    # This implies we want to quote *above* mid? No, bid is below mid.
    # Negative depth means the optimal bid price is above mid.
    # In this model, we can only quote below mid (positive depth).
    # So we should clamp to a small positive value or 0?
    # The constraint is > -3/k.
    # So we can clamp to -3/k + epsilon.
    
    min_depth = -3.0 / k + 1e-9
    max_depth = 100.0 / k
    
    bid_depths = np.clip(bid_depths, min_depth, max_depth)
    ask_depths = np.clip(ask_depths, min_depth, max_depth)
    
    # Handle boundary inventory: no bid at Q, no ask at -Q
    at_max_inventory = (inventory == Q)
    at_min_inventory = (inventory == -Q)
    
    bid_depths[at_max_inventory] = np.inf
    ask_depths[at_min_inventory] = np.inf
    
    # Ensure no NaN or -inf
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths == -np.inf), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths == -np.inf), np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
