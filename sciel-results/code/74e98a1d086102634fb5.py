# EVOLVE-BLOCK-START
import numpy as np


def quote(t, inventory, mid, params):
    """Return (bid_depths, ask_depths), float arrays of shape (N,), distances
    below/above mid to quote this step; np.inf means no quote."""
    sigma = float(params["sigma"])
    A = float(params["A"])
    k = float(params["k"])
    gamma = float(params["gamma"])
    T = float(params["T"])
    Q = int(params["Q"])

    n = len(inventory)
    q = np.asarray(inventory, dtype=np.float64)

    # Time remaining
    tau = np.clip(T - float(t), 0.0, T)

    # Reservation price shift r
    # r = gamma * sigma^2 * q * tau
    # If q > 0, r > 0. We want to sell (ask aggressive, bid passive).
    r = gamma * sigma * sigma * q * tau

    inv_k = 1.0 / k
    
    # Optimal depths
    # d_b = r + 1/k
    # d_a = 1/k - r
    bid = r + inv_k
    ask = inv_k - r

    # Validity constraints
    lo = -3.0 / k
    hi = float(Q)
    
    # Clip to valid range
    bid = np.clip(bid, lo, hi)
    ask = np.clip(ask, lo, hi)

    # Boundary constraints
    # No bid fill at q = Q
    bid = np.where(q >= Q, np.inf, bid)
    
    # No ask fill at q = -Q
    ask = np.where(q <= -Q, np.inf, ask)
    
    # Final safety check
    bid = np.where(np.isfinite(bid) & (bid >= lo), bid, np.inf)
    ask = np.where(np.isfinite(ask) & (ask >= lo), ask, np.inf)
    
    return bid, ask
# EVOLVE-BLOCK-END
