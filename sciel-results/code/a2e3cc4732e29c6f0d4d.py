<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def quote(t, inventory, mid, params):
    """Seed: quote both sides at the risk-neutral depth 1/k, ignoring inventory."""
    depth = np.full(len(inventory), 1.0 / params["k"])
    return depth, depth.copy()
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0


def quote(t, inventory, mid, params):
    """
    Inventory-aware market making using Avellaneda-Stoikov style reservation price shift.
    Adjusts bid and ask depths based on current inventory to manage risk.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    
    # Time remaining
    dt_rem = T - t
    
    # Inventory risk adjustment (Reservation Price Shift)
    # R = gamma * sigma^2 * q * (T - t)
    # If q > 0 (long), R > 0. We want to sell.
    # Bid should be further away (wider depth), ask closer (narrower depth)
    # If q < 0 (short), R < 0. We want to buy.
    # Bid should be closer (narrower depth), ask further away (wider depth)
    
    q = np.asarray(inventory, dtype=np.float64)
    
    # Base depth
    base_depth = 1.0 / k
    
    # Reservation price shift
    R = gamma * sigma**2 * q * dt_rem
    
    # Adjust depths:
    # bid_depth = base_depth - R (if R > 0, bid moves down = wider)
    # ask_depth = base_depth + R (if R > 0, ask moves up = wider)
    # But we need positive depths. Clip to minimum value.
    
    min_depth = 1e-6  # Small positive value to avoid invalid depths
    
    bid_depth = base_depth - R
    ask_depth = base_depth + R
    
    # Ensure depths are positive and finite
    bid_depth = np.maximum(bid_depth, min_depth)
    ask_depth = np.maximum(ask_depth, min_depth)
    
    # Handle boundary conditions: no bid fill at q = Q, no ask fill at q = -Q
    Q = params["Q"]
    bid_depth[q >= Q] = np.inf
    ask_depth[q <= -Q] = np.inf
    
    return bid_depth, ask_depth
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

BET: {mechanism: "Inventory-aware reservation price shift (Avellaneda-Stoikov) corrects the seed's inventory-blindness, significantly improving objective by aligning quotes with inventory risk. Measured: 1 -> 1.232. (strategy of a program that improved island 2)", q: 0.85, delta: [0.2, 0.4]}
