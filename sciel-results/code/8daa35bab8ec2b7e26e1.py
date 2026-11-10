"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def quote(t, inventory, mid, params):
    """
    Inventory-aware market making policy.
    
    Uses skew and spread adjustments based on inventory level and time remaining.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    Q = params['Q']
    
    n_paths = len(inventory)
    inv = np.asarray(inventory, dtype=np.float64)
    
    # Time remaining fraction (1 at start, 0 at end)
    t_rem = max(0.0, 1.0 - t / T)
    
    # Base depth: risk-neutral for a single side
    # Fill prob ≈ A * exp(-k*d) * dt for small d
    # Optimal base balances fill frequency vs. depth earned
    dt = T / params['steps']
    base_depth = 1.0 / k  # starting point
    
    # Inventory skew: shift quotes to encourage inventory reduction
    # If inv > 0, we want to sell more -> make ask tighter, bid wider
    # Skew magnitude proportional to inventory and risk aversion
    skew_coeff = gamma * sigma * sigma * dt / (2.0 * A * np.exp(-k * base_depth) * k)
    # Clamp skew to reasonable range
    skew = np.clip(skew_coeff * inv, -2.0 / k, 2.0 / k)
    
    # Spread widening: increase both sides when |inventory| is large
    # This reduces fill probability to control inventory
    spread_widen = 0.5 * np.abs(inv) / Q * (1.0 / k)
    # Scale with time: more aggressive when less time remaining
    spread_widen *= (1.0 + 0.5 * (1.0 - t_rem))
    
    # Combine: bid depth = base - skew + spread_widen, ask depth = base + skew + spread_widen
    # When inv > 0: skew > 0, so bid gets wider (base + spread - skew could be small), ask gets tighter
    # Wait, let me reconsider the sign convention.
    
    # Bid depth: distance below mid. Larger depth = less likely to fill.
    # Ask depth: distance above mid. Larger depth = less likely to fill.
    
    # If we have positive inventory, we want to reduce it by selling (ask fills).
    # So we want ask to be tighter (smaller depth) and bid to be wider (larger depth).
    # skew > 0 when inv > 0:
    #   bid_depth = base_depth + spread_widen - skew  (wider bid, harder to buy)
    #   ask_depth = base_depth + spread_widen + skew  (tighter ask? No, that's wider)
    
    # Let me redefine: skew shifts the midpoint of the quote
    # bid_depth = base_depth + spread_widen - skew
    # ask_depth = base_depth + spread_widen + skew
    # When inv > 0, skew > 0: bid_depth decreases (tighter bid, easier to buy - BAD)
    # That's wrong. Let me flip the sign.
    
    # Correct: when inv > 0, we want:
    #   bid_depth to INCREASE (wider, less likely to buy)
    #   ask_depth to DECREASE (tighter, more likely to sell)
    # So: bid_depth = base + spread + |skew|, ask_depth = base + spread - |skew|
    # But we need to handle negative inventory too.
    
    # Let skew = c * inv where c > 0
    # bid_depth = base_depth + spread_widen + skew
    # ask_depth = base_depth + spread_widen - skew
    # When inv > 0, skew > 0: bid wider, ask tighter. Good.
    # When inv < 0, skew < 0: bid tighter, ask wider. Good.
    
    bid_depths = base_depth + spread_widen + skew
    ask_depths = base_depth + spread_widen - skew
    
    # Ensure minimum depth to avoid infinite fill probability
    min_depth = 1e-4
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    # At inventory limits, don't quote on the side that would increase inventory
    # No bid fill at q = Q (can't buy more)
    bid_depths[inv >= Q] = np.inf
    # No ask fill at q = -Q (can't sell more)
    ask_depths[inv <= -Q] = np.inf
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
