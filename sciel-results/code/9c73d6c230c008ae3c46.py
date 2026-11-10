import numpy as np
import time

def heuristics(demands, capacity):
    """
    Heuristic matrix for ACO bin packing: attractiveness of placing item j after item i.
    
    Uses a product-to-waste-squared ratio that rewards tight packing and large items.
    The squared denominator strongly prioritizes pairs that leave minimal waste,
    which helps the colony find dense bin configurations.
    """
    start_time = time.time()

    d = np.asarray(demands, dtype=np.float64)
    n = len(d)

    if n == 0:
        return np.empty((0, 0))

    # Pairwise sums and products
    sums = d[:, None] + d[None, :]
    products = d[:, None] * d[None, :]

    # Feasibility mask
    fits = sums <= capacity

    # Waste = capacity - sum (non-negative where fits)
    waste = capacity - sums

    # Safe denominator: for fitting pairs, waste + 1 >= 1; for non-fitting, use 1.0
    # Squaring the denominator amplifies the penalty for wasted space
    safe_waste_plus_one = np.where(fits, waste + 1.0, 1.0)
    denom = safe_waste_plus_one ** 2

    # Score: product / (waste + 1)^2 -- higher for tight, large-item pairs
    raw_scores = products / denom

    # Mask non-fitting pairs to 1e-9
    H = np.where(fits, raw_scores, 1e-9)

    # Sanitize: replace any non-finite values
    if not np.all(np.isfinite(H)):
        H = np.where(np.isfinite(H), H, 1e-9)

    # Shape check
    if H.shape != (n, n):
        raise ValueError("Wrong shape")

    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 290:
        return H

    return H
