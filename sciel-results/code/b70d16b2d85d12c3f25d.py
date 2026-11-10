"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    pair_sum = demands[:, None] + demands[None, :]
    fits = pair_sum <= capacity
    gap = capacity - pair_sum
    # Inverse gap: reward pairs that leave little space
    # Add 1 to gap to avoid division by zero and keep values bounded
    h = np.where(fits, 1.0 / (gap + 1.0), 1e-9)
    return h
# EVOLVE-BLOCK-END
