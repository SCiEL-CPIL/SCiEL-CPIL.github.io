import numpy as np

def heuristics(demands, capacity):
    demands = np.array(demands, dtype=float)
    n = len(demands)
    
    # Pair sums: demands[i] + demands[j]
    pair_sums = demands[:, None] + demands[None, :]
    
    # Valid mask: pair_sums <= capacity
    valid = pair_sums <= capacity
    
    # Complementarity score: 1 / (1 + waste)
    # This rewards pairs that fill the bin more tightly (smaller waste).
    # It is more non-linear (greedy) than the linear d_j / (C - d_i) heuristic.
    waste = capacity - pair_sums
    
    eta = np.where(valid, 1.0 / (1.0 + waste), 1e-9)
    
    return eta
