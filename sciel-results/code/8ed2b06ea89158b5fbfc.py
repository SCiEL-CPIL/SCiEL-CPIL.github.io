import numpy as np
def heuristics(demands, capacity):
    n = len(demands)
    d = np.array(demands, dtype=float)
    pair_sum = d[:, None] + d[None, :]
    fits = pair_sum <= capacity
    # For fitting pairs, use inverse gap to strongly favor tight packing
    gap = capacity - pair_sum  # >= 0 for fitting pairs
    score = np.where(fits, 1.0 / (gap + 1.0), 1e-9)
    return score
