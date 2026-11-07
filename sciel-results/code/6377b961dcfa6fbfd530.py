import numpy as np
import time

# Global cache for precomputed constants per regime
_params_cache = {}

def _get_constants(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _params_cache:
        sigma = params['sigma']
        A = params['A']
        k = params['k']
        gamma = params['gamma']
        T = params['T']
        steps = params['steps']
        Q = params['Q']
        
        dt = T / steps
        
        # Effective risk aversion per step
        # Objective term: (gamma / 2) * sigma^2 * q^2 * dt
        # Standard A-S form: (gamma_as / 2) * q^2
        # So gamma_as = gamma * sigma^2 * dt
        gamma_as = gamma * sigma**2 * dt
        
        # Base half-spread (depth from mid for q=0)
        # Formula: h = (1/gamma_as) * ln(1 + gamma_as/k)
        # If gamma_as is very small, h ~ 1/k.
        # If gamma_as is large, h ~ (1/gamma_as) * ln(gamma_as/k).
        if gamma_as > 1e-9:
            base_depth = (1.0 / gamma_as) * np.log1p(gamma_as / k)
        else:
            base_depth = 1.0 / k
            
        # Skew coefficient
        # Formula: alpha = (gamma_as / k) * ln(1 + gamma_as/k)
        if gamma_as > 1e-9:
            skew_coeff = (gamma_as / k) * np.log1p(gamma_as / k)
        else:
            skew_coeff = 0.0
            
        # Clamp base depth to valid range
        min_depth = -3.0 / k + 1e-6
        base_depth = max(base_depth, min_depth)
        
        _params_cache[key] = {
            'base_depth': base_depth,
            'skew_coeff': skew_coeff,
            'min_depth': min_depth,
            'k': k
        }
        
    return _params_cache[key]

def quote(t, inventory, mid, params):
    # Get precomputed constants
    consts = _get_constants(params)
    base_depth = consts['base_depth']
    skew_coeff = consts['skew_coeff']
    min_depth = consts['min_depth']
    k = consts['k']
    
    # Inventory is an array of ints
    inv = np.asarray(inventory, dtype=np.float64)
    
    # Calculate depths
    # bid_depth = base + skew * q
    # ask_depth = base - skew * q
    # Note: If q > 0 (long), we want to sell more (increase ask depth? No, decrease ask depth to hit more? 
    # Wait. 
    # If I am long (q>0), I want to reduce inventory.
    # I want to fill asks (sell) more often. So Ask Depth should be SMALLER (closer to mid).
    # I want to fill bids (buy) less often. So Bid Depth should be LARGER (further from mid).
    # 
    # Let's check the sign in A-S.
    # A-S Bid: p_b - w(q)
    # A-S Ask: p_a + w(q)
    # Where w(q) is positive for q>0.
    # Bid price is lower (further below mid) -> Bid Depth is LARGER.
    # Ask price is higher (further above mid) -> Ask Depth is LARGER? 
    # No, wait.
    # Mid is M.
    # Bid = M - d_b.
    # Ask = M + d_a.
    # If q > 0 (long):
    # We want to sell. So we want Ask to be hit.
    # To make Ask more likely to be hit, it should be CLOSER to Mid. So d_a should be SMALLER.
    # To make Bid less likely to be hit, it should be FURTHER from Mid. So d_b should be LARGER.
    #
    # So:
    # d_b = base + alpha * q  (alpha > 0)
    # d_a = base - alpha * q  (alpha > 0)
    #
    # Let's verify with A-S notation.
    # A-S defines spread around a reservation price.
    # Reservation price = Mid - w(q).
    # If q > 0, Reservation Price < Mid.
    # Bid = Res - h = Mid - w(q) - h.
    # Depth d_b = w(q) + h. (Larger)
    # Ask = Res + h = Mid - w(q) + h.
    # Depth d_a = w(q) - h. (Smaller, assuming w(q) < h? Or just relative change)
    # Actually, if w(q) is large, Ask could be below Mid? 
    # The model assumes depths are positive distances.
    # If d_a becomes negative, we are quoting Ask below Mid, which is crossing the spread.
    # The model allows d > -3/k.
    # So yes, d_a = base - skew * q is correct direction.
    
    bid_depths = base_depth + skew_coeff * inv
    ask_depths = base_depth - skew_coeff * inv
    
    # Ensure depths are within valid bounds
    # Values below -3/k are invalid.
    # Also, typically we don't want negative depths that are too large (crossing too much), 
    # but the problem allows down to -3/k.
    
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    # Ensure no NaN/Inf
    bid_depths = np.nan_to_num(bid_depths, nan=base_depth, posinf=base_depth, neginf=min_depth)
    ask_depths = np.nan_to_num(ask_depths, nan=base_depth, posinf=base_depth, neginf=min_depth)
    
    return bid_depths, ask_depths
